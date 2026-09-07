import os
os.environ['OMP_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'

import argparse
from configs import cfg
import torch
import torch.nn as nn
import os.path as osp
from tqdm import tqdm
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

from utils.timestamp import what_time_is_it
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.scheduler import create_scheduler
from utils.seeds import fix_every_seed
import logging

from models import make_UNet

from datasets.loaders import make_dataloader
from processors.processor import do_test

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from lion_pytorch import Lion

def Evaluate(model,
             test_loader_dict,
             accelerator: Accelerator,
             args=None
    ):
    """ 
    ### Functions:
    - a PnP function, to be embedded into do_train, not used in indepent testing.
    """
    paths = [None]
    if accelerator.is_main_process:
        time = what_time_is_it()
        log_dir_name = f"{time}-{args.desc}"
        paths[0] = osp.join(args.eval_root, log_dir_name)

    paths = broadcast_object_list(object_list=paths, from_process=0)
    log_dir = paths[0]

    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        Logger = setup_logger(
            name="RadioMapEstim.eval",
            save_dir=log_dir,
            if_train=False # test_log.txt
        )
        Logger.propagate = False
    else:
        Logger = logging.getLogger(
            f"eval_silent_rank_{accelerator.process_index}"
        )
        Logger.disabled = True
    accelerator.wait_for_everyone()

    device = accelerator.device
    evaluator = Evaluator(device=accelerator.device)
    Logger.info(f"Using {device}")
    Logger.info(f"Accelerator process numbers: {accelerator.num_processes}")
    Logger.info(f"Evaluator is successfully initialized.")

    model.eval()
    for geo_type in cfg.INPUT.HYBRID_GEO:
        if cfg.MODEL.SINGLE_HEIGHT:
            height_indices = [int(cfg.CONDITION.TARGET_HEIGHT)]
        else:
            height_indices = range(len(cfg.CONDITION.HEIGHT_LIST))

        for height_idx in height_indices:
        #for height_idx in range(len(cfg.CONDITION.HEIGHT_LIST)):
            h = cfg.CONDITION.HEIGHT_LIST[height_idx] * 10
            f_idx = int(cfg.CONDITION.TARGET_FREQUENCY[1:])
            f = cfg.CONDITION.FREQUENCIES_LIST[f_idx] / 100

            name = f"{geo_type}_{h}m_{f}GHz"
            result_file = osp.join(log_dir, f"{name}.txt")

            evaluator.reset()
            results = do_test(
                model=model,
                accelerator=accelerator,
                test_loader=test_loader_dict[geo_type],
                evaluator=evaluator,
                cfg=cfg,
                height_idx=height_idx,
                Logger=Logger,
                geo_type=geo_type
            )

            results: dict
            if accelerator.is_main_process:
                with open(result_file, "w") as f:
                    f.write(f"Results for {name}:\n")
                    Logger.info(f"Results for {name}:")
                    for k, v in results.items():
                        f.write(f"{k}: {v:.4f}\n")
                        Logger.info(f"{k}: {v:.4f}")
            accelerator.wait_for_everyone()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RME")
    parser.add_argument("--config_file", default="./configs/SpectrumNet.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--eval_root", default="./_EVALs", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument("--desc", default="say_something", type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    accelerator = Accelerator()
    device = accelerator.device

    paths = [None]
    if accelerator.is_main_process:
        time = what_time_is_it()
        log_dir_name = f"{time}-{args.desc}"
        paths[0] = osp.join(args.eval_root, log_dir_name)

    paths = broadcast_object_list(object_list=paths, from_process=0)
    log_dir = paths[0]

    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        Logger = setup_logger(
            name="RadioMapEstim.eval",
            save_dir=log_dir,
            if_train=False # test_log.txt
        )
        Logger.propagate = False
    else:
        Logger = logging.getLogger(
            f"eval_silent_rank_{accelerator.process_index}"
        )
        Logger.disabled = True
    accelerator.wait_for_everyone()

    evaluator = Evaluator(device=accelerator.device, sync_on_compute=False)
    Logger.info(f"Using {device}")
    Logger.info(f"Accelerator process numbers: {accelerator.num_processes}")
    Logger.info(f"Evaluator is successfully initialized.")

    model = make_UNet(cfg=cfg, Logger=Logger)
    
    train_loader, test_loader_dict = make_dataloader(cfg=cfg, Logger=Logger)

    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    model_dict = model.state_dict()

    for k,v in state_dict.items():
        if k in model_dict and v.shape != model_dict[k].shape:
            Logger.info(f'plz check {k}: loaded {v.shape}, model {model_dict[k].shape}.')

    model.load_state_dict(state_dict, strict=True)
    model = accelerator.prepare(model)
    Logger.info(f"Checkpoint has been successfully loaded from {args.ckpt_path}.")
    model = model.to(device)
    model.eval()

    test_loader_dict = {
        geo_type: accelerator.prepare_data_loader(loader) \
        for geo_type, loader in test_loader_dict.items()
    }

    for geo_type in cfg.INPUT.HYBRID_GEO:
        if cfg.MODEL.SINGLE_HEIGHT:
            height_indices = [int(cfg.CONDITION.TARGET_HEIGHT)]
        else:
            height_indices = range(len(cfg.CONDITION.HEIGHT_LIST))
            
        for height_idx in height_indices:
        #for height_idx in range(len(cfg.CONDITION.HEIGHT_LIST)):
            h = cfg.CONDITION.HEIGHT_LIST[height_idx] * 10
            f_idx = int(cfg.CONDITION.TARGET_FREQUENCY[1:])
            f = cfg.CONDITION.FREQUENCIES_LIST[f_idx] / 100

            name = f"{geo_type}_{h}m_{f}GHz"
            result_file = osp.join(log_dir, f"{name}.txt")

            evaluator.reset()
            results = do_test(
                model=model,
                accelerator=accelerator,
                test_loader=test_loader_dict[geo_type],
                evaluator=evaluator,
                cfg=cfg,
                height_idx=height_idx,
                Logger=Logger,
                geo_type=geo_type
            )

            if accelerator.is_main_process:
                results: dict
                with open(result_file, "w") as f:
                    f.write(f"Results for {name}:\n")
                    Logger.info(f"Results for {name}:")
                    for k, v in results.items():
                        f.write(f"{k}: {v:.4f}" + "\n")
                        Logger.info(f"{k}: {v:.4f}")
            accelerator.wait_for_everyone()

    Logger.info(f"Evaluation finished.")
    accelerator.end_training()

                     
                 
