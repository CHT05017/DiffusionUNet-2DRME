import os
os.environ['OMP_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'

import argparse
from configs import cfg
import torch
import torch.nn as nn
import os.path as osp
from tqdm import tqdm
import logging

from utils.timestamp import what_time_is_it
from utils.logger import setup_logger
from utils.scheduler import create_scheduler
from utils.seeds import fix_every_seed
from utils.z00 import create_z00_info_txt

from models import make_UNet
from models.diffusion.ddpm import DDPM

from datasets.loaders import make_dataloader
from processors.processor import do_train
from test import Evaluate

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from lion_pytorch import Lion

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RME")
    parser.add_argument("--config_file", default="./configs/SpectrumNet.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--log_root", default="./_LOGs", type=str)
    parser.add_argument("--ckpt_root", default="./_CKPTs", type=str)
    parser.add_argument("--eval_root", default="./_EVALs", type=str)
    parser.add_argument("--desc", default="say_something", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    fix_every_seed(seed=cfg.SOLVER.SEEDS)
    accelerator = Accelerator()
    device = accelerator.device

    paths = [None, None]
    if accelerator.is_main_process:
        time = what_time_is_it()
        log_dir_name = f"{time}-{args.desc}"
        paths[0] = osp.join(args.log_root, log_dir_name)
        paths[1] = osp.join(args.ckpt_root, log_dir_name)

    paths = broadcast_object_list(object_list=paths, from_process=0)
    log_dir, ckpt_dir = paths

    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
    
        Logger = setup_logger(
            name="RadioMapEstim.train",
            save_dir=log_dir,
            if_train=True # train_log.txt
        )
    else:
        Logger = logging.getLogger(name=f"train_silent_rank_{accelerator.process_index}")
        Logger.disabled = True
    accelerator.wait_for_everyone()

    Logger.info(f"Using {device}")
    Logger.info(f"Accelerator process numbers: {accelerator.num_processes}")

    if accelerator.is_main_process:
        create_z00_info_txt(cfg=cfg, Logger=Logger)
    accelerator.wait_for_everyone()

    model = make_UNet(cfg=cfg, Logger=Logger)
    train_loader, test_loader_dict = make_dataloader(cfg=cfg, Logger=Logger)

    loss = nn.MSELoss()

    # optimizer = Lion(
    #    params=model.parameters(),
    #    lr=cfg.SOLVER.LR,
    #    weight_decay=cfg.SOLVER.WEIGHT_DECAY
    #)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.SOLVER.LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )

    scheduler = create_scheduler(cfg=cfg, optimizer=optimizer)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    do_train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=loss,
        accelerator=accelerator,
        scheduler=scheduler,
        Logger=Logger,
        ckpt_dir=ckpt_dir,
        cfg=cfg
    )

    Logger.info(f"Training is finished.")

    test_loaders = {geo_type: accelerator.prepare_data_loader(loader) \
                        for geo_type, loader in test_loader_dict.items()}
    Logger.info(f"Entering model evaluation.")

    Evaluate(
        model=model,
        test_loader_dict=test_loaders,
        args=args,
        accelerator=accelerator
    )

    Logger.info(f"Model evaluation finished. Everything done.")
    accelerator.end_training()



    



