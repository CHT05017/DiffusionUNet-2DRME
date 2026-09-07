import torch
import torch.nn as nn
import random
import os
import os.path as osp

from accelerate import Accelerator
from utils.logger import setup_logger
from models.diffusion.ddpm import DDPM
from models.diffusion.ddim import DDIM
from models.physics_model import RSS_project
from models.physics_model import estimate_tx_map
from models.utils import height_embed

from utils.metre import AverageMeter
from utils.timestamp import what_time_is_it

from tqdm import tqdm
import time as tm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def do_train(
        model,
        optimizer,
        train_loader,
        accelerator: Accelerator,
        scheduler,
        loss_fn,
        Logger, # logger for train
        ckpt_dir,
        cfg=None
):
    """
    ### Function:
    - Train and Eval, combined together.
    """
    assert 1 <= cfg.MODEL.DDIM_STEPS <= cfg.MODEL.DIFFUSION_STEPS
    loss_meter = AverageMeter()
    height = cfg.CONDITION.TARGET_HEIGHT
    device = next(model.parameters()).device

    ddpm = DDPM(
        device=device,
        n_steps=cfg.MODEL.DIFFUSION_STEPS
    )

    best_loss = float("inf")
    for epoch in range(1, cfg.SOLVER.EPOCHS + 1):
        tic = tm.time()
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, batch in tqdm(enumerate(train_loader), 
                                    total=len(train_loader), desc=f"Ep-{epoch}/{cfg.SOLVER.EPOCHS}",
                                    disable=not accelerator.is_main_process):
            radiomap, sampled_map, sampled_mask, building, terrain, frequency, env, pointcloud = batch

            B = radiomap.shape[0]

            if cfg.DEBUG.OVERFIT:
                height_idx = cfg.DEBUG.HEIGHT_IDX
            elif height == "Hybrid":
                height_idx = random.randint(0, 2)
            else:
                height_idx = int(height)

            # if height == "Hybrid":
            #    height_idx = random.randint(0, 2)
            #else:
            #    height_idx = int(height)

            physics_rm, physics_mask = RSS_project(
                sampled_map=sampled_map.cpu(),
                sampled_mask=sampled_mask.cpu(),
                building=building.cpu(),
                env=env,
                target_height_idx=height_idx,
                cfg=cfg,
                pointcloud=pointcloud.cpu()
            )

            # NOTE temp notes
            if epoch == 1 and n_iter == 0:
                counts = sampled_mask[0].sum(dim=(1, 2))

                expected = torch.zeros_like(counts)
                expected[height_idx] = cfg.INPUT.NUM_SAMPLE

                observed = sampled_map[:, height_idx:height_idx + 1].cpu()
                observed_mask = sampled_mask[:, height_idx:height_idx + 1].cpu().bool()

                Logger.info(
                    f"Single-height baseline: "
                    f"height_idx={height_idx}, "
                    f"sample_counts={counts.tolist()}, "
                    f"steps_per_epoch={len(train_loader)}"
                )
                assert torch.equal(counts, expected)
                assert height_idx == int(cfg.CONDITION.TARGET_HEIGHT)
                assert torch.equal(physics_rm[observed_mask], observed[observed_mask])

            # tx_map = estimate_tx_map(
            #    sampled_map=sampled_map.cpu(),
            #    sampled_mask=sampled_mask.cpu(),
            #    building=building.cpu(),
            #    target_height_idx=height_idx,
            #    cfg=cfg
            #)

            building = building.to(device)
            radiomap = radiomap.to(device)
            sampled_map = sampled_map.to(device)
            terrain = terrain.to(device)
            physics_rm = physics_rm.to(device)
            physics_mask = physics_mask.to(device)
            #tx_map = tx_map.to(device)


            viz_here = False
            if False:
                if accelerator.is_main_process:
                    plt.imsave(
                        f"./{n_iter}_Tx.png",
                        tx_map[0, 0].detach().cpu().numpy(),
                        cmap="viridis"
                    )
                    plt.imsave(
                        f"./{n_iter}_GT.png",
                        radiomap[0, 0].detach().cpu().numpy(),
                        cmap="viridis"
                    )
                accelerator.wait_for_everyone()

            canvas = radiomap[:, height_idx: height_idx+1]

            time = torch.randint(
                low=0,
                high=cfg.MODEL.DIFFUSION_STEPS,
                size=(B, ),
                device=device
            ) # (B, )

            epsilon = torch.randn_like(
                input=canvas,
                device=device
            )

            x_t = ddpm.sample_forward(
                x=canvas,
                t=time,
                eps=epsilon
            )

            height_map = height_embed(
                height_idx=height_idx,
                cur_batch_size=B,
                device=device,
                cfg=cfg
            )

            normed_bd, normed_terr = map(
                lambda x: x * 2 - 1,
                (building[:, height_idx: height_idx + 1], terrain)
            )


            viz = False
            if viz:
                names = ["building", "terrain", "height_map", "radiomap"]
                variables = [normed_bd[0, 0], normed_terr[0, 0], height_map[0, 0], radiomap[0, 0]]
                for name, var in zip(names, variables):                
                    plt.imsave(
                            osp.join(f"./{name}.png"),
                                var.detach().cpu().numpy(),
                                cmap="viridis"
                            )
                    

            input = torch.cat(
                tensors=(x_t, physics_rm, physics_mask, height_map, normed_bd, normed_terr),
                dim=1
            )

            eps_pred = model(input, time)
            loss = loss_fn(eps_pred, epsilon)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            loss_meter.update(loss.item(), B)

            # if n_iter % cfg.SOLVER.LOG_ITER == 0:
            #    Logger.info(f"Current loss: {loss_meter.avg}")

        loss_stats = torch.tensor(
            [loss_meter.sum, loss_meter.count],
            dtype=torch.float64,
            device=accelerator.device
        )

        loss_stats = accelerator.reduce(
            loss_stats,
            reduction="sum"
        )
            
        epoch_loss = (loss_stats[0] / loss_stats[1]).item()

        tok = tm.time()
        elapsed = tok - tic
        Logger.info(f"Epoch {epoch} finished in {elapsed:.2f}s")
        Logger.info(f"Epoch {epoch} Avg Loss: {epoch_loss:.6f}")
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch     

            Logger.info(f"Best CKPT is found at Epoch {best_epoch}.")
            Logger.info(f"Best CKPT is stored at {ckpt_dir}.")

            accelerator.wait_for_everyone()
            state_dict = accelerator.get_state_dict(model, unwrap=True)

            if accelerator.is_main_process:
                ckpt_path = osp.join(ckpt_dir, f"best.pth")
                accelerator.save(state_dict, ckpt_path)
               

        if epoch == cfg.SOLVER.EPOCHS:
            Logger.info(f"Training finished.")
            accelerator.wait_for_everyone()
            last_state_dict = accelerator.get_state_dict(model, unwrap=True)

            if accelerator.is_main_process:
                last_ckpt_path = osp.join(ckpt_dir, f"latest.pth")
                accelerator.save(last_state_dict, last_ckpt_path)

            accelerator.wait_for_everyone()


from utils.metrics import Evaluator
def do_test(
        model,
        test_loader,
        geo_type,
        height_idx,
        evaluator: Evaluator,
        accelerator: Accelerator,
        cfg=None,
        Logger=None,
):
    def _rerank(
            physics_rm,
            imgs_predict):
        imgs_predict = imgs_predict.clamp(min=-1, max=1)
        mask = physics_rm == -1

        refined_imgs = torch.masked_fill(input=imgs_predict, mask=mask, value=-1) # (B, 1, H, W)
        imgs_prior = physics_rm # (B, 1, H, W)

        mse = lambda x, y: (x - y).square().mean(dim=(1, 2, 3))
        errs = mse(imgs_prior, refined_imgs) # (B, )

        best_img_idx = torch.argmin(input=errs)
        best_img = imgs_predict[best_img_idx: best_img_idx + 1] # (1, 1, H, W)

        # [-1, 1] -> [0, 1]
        best_img = (best_img + 1) / 2

        return best_img


    model.eval()

    predicts_list = []
    targets_list = []
    device = accelerator.device 
    ddim = DDIM(device=device, n_steps=cfg.MODEL.DIFFUSION_STEPS)

    rerank_num = cfg.TEST.RERANK_FORCE * 1
    # if cfg.DEBUG.SINGLE_HEIGHT_BASELINE:
    #    assert rerank_num == 1
    assert rerank_num >= 1
    assert cfg.TEST.BATCH_SIZE == 1, f"Suggested batch_size=1 when testing."

    accelerator.wait_for_everyone()
    with torch.no_grad():
        for idx, (radiomap, sampled_map, sampled_mask, building, terrain, freq, env, pointcloud) in tqdm(enumerate(test_loader), desc=f"{geo_type}",
                                                        disable=not accelerator.is_main_process,
                                                        total=len(test_loader)):
            building, terrain = \
                building.repeat(rerank_num, 1, 1, 1), terrain.repeat(rerank_num, 1, 1, 1)

            physics_rm, physics_mask = RSS_project(
                sampled_map=sampled_map.cpu(),
                sampled_mask=sampled_mask.cpu(),
                building=building[0: 1].cpu(),
                env=env,
                target_height_idx=height_idx,
                cfg=cfg,
                pointcloud=pointcloud.cpu()
            )
            physics_rm = physics_rm.repeat(rerank_num, 1, 1, 1)
            physics_mask = physics_mask.repeat(rerank_num, 1, 1, 1)

            radiomap, sampled_map, building, terrain, physics_rm, physics_mask = map(
                lambda x: x.to(device), [radiomap, sampled_map, building, terrain, physics_rm, physics_mask]
            )
            height_map = height_embed(
                height_idx=height_idx,
                cur_batch_size=cfg.TEST.BATCH_SIZE,
                device=device,
                cfg=cfg,
                is_test=True
            )

            # for GT, [-1, 1] -> [0, 1]
            groundtruth = (radiomap[:, height_idx: height_idx + 1] + 1) / 2

            # for input condition, [0, 1] -> [-1, 1]
            # height_map is natually [-1, 1] due to embedding
            normed_bd, normed_terr = map(
                lambda x: x * 2 - 1,
                (building[:, height_idx: height_idx + 1], terrain)
            )
            input = torch.cat(
                tensors=(physics_rm, physics_mask, height_map, normed_bd, normed_terr),
                dim=1
            )

            sample_shape = (rerank_num, 1, cfg.INPUT.MAP_SHAPE[0], cfg.INPUT.MAP_SHAPE[1])

            if cfg.DEBUG.SINGLE_HEIGHT_BASELINE:
                generator = torch.Generator(device=device)
                generator.manual_seed(cfg.SOLVER.SEEDS + 100000 + idx * accelerator.num_processes + accelerator.process_index)
                img_or_shape = torch.randn(sample_shape, generator=generator, device=device)
            else:
                img_or_shape = sample_shape

            imgs_predict = ddim.sample_backward(
                img_or_shape=img_or_shape,
                net=model,
                condition_without_noise=input,
                device=device,
                ddim_step=cfg.MODEL.DDIM_STEPS,
                print_tqdm=False
            )


            """
            raw_pred = imgs_predict[:1]
            pred = (raw_pred.clamp(-1, 1) + 1) / 2
            gt = groundtruth

            obs_mask = physics_mask[:1].to(device).bool()
            obs_gt = (physics_rm[:1].to(device) + 1) / 2

            full_error = pred - gt
            obs_error = pred[obs_mask] - obs_gt[obs_mask]

            Logger.info(
                f"sample={idx}, "
                f"raw_range=[{raw_pred.min().item():.4f}, "
                f"{raw_pred.max().item():.4f}], "
                f"clip_high={(raw_pred > 1).float().mean().item():.4f}, "
                f"pred_mean={pred.mean().item():.4f}, "
                f"gt_mean={gt.mean().item():.4f}, "
                f"full_bias={full_error.mean().item():.4f}, "
                f"obs_bias={obs_error.mean().item():.4f}, "
                f"obs_mae={obs_error.abs().mean().item():.4f}"
            )
            """


            if cfg.DEBUG.SINGLE_HEIGHT_BASELINE:
                # best_img = imgs_predict[:1].clamp(-1, 1) + 1
                # best_img = best_img / 2
                obs_mask = physics_mask.bool()
                obs_value = physics_rm

                candidate_error = (
                    (imgs_predict - obs_value).square()
                    * obs_mask
                ).sum(dim=(1, 2, 3)) / obs_mask.sum(dim=(1, 2, 3))

                best_idx = candidate_error.argmin()
                best_img = (
                    imgs_predict[best_idx:best_idx + 1].clamp(-1, 1) + 1
                ) / 2
                Logger.info(
                        f"sample={idx}, "
                        f"candidate_obs_mse_min={candidate_error.min().item():.6f}, "
                        f"median={candidate_error.median().item():.6f}, "
                        f"max={candidate_error.max().item():.6f}"
                    )
            else:
                best_img = _rerank(
                    physics_rm=physics_rm,
                    imgs_predict=imgs_predict
                )

            vis = True
            if vis:
                if accelerator.is_main_process:
                    root = f"./_PICs/{geo_type}"
                    if not osp.exists(root):
                        os.makedirs(root, exist_ok=True)
                    plt.imsave(
                        osp.join(root, f"{geo_type}_height{height_idx}_{idx}_best_img.png"),
                        best_img[0, 0].detach().cpu().numpy(),
                        cmap="viridis", vmin=0, vmax=1,
                    )
                    plt.imsave(
                        osp.join(root, f"{geo_type}_height{height_idx}_{idx}_GroundTruth.png"),
                        groundtruth[0, 0].detach().cpu().numpy(),
                        cmap="viridis", vmin=0, vmax=1,
                    )
            


            # BUG DDP Eval evaluator.update(preds=best_img.float(),
            # BUG                 targets=groundtruth.float())
            # Only the main thread calculates the metrics
            # Other threads only collect the data
            all_preds, all_targets = accelerator.gather_for_metrics(input_data=(best_img.float(), groundtruth.float()))
            if accelerator.is_main_process:
                evaluator.update(preds=all_preds, targets=all_targets)

    results = None

    if accelerator.is_main_process:
        results = evaluator.compute()

    accelerator.wait_for_everyone()
    return results



            
        




