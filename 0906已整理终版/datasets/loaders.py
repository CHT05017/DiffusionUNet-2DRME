import os
import os.path as osp
import sys
import numpy as np

from datasets.utils import preprocessor, collate_fn
from datasets.datasets.SpectrumNet import SpectrumNet
from torch.utils.data import DataLoader, DistributedSampler

current_dir = osp.dirname(osp.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def make_dataloader(cfg, Logger):
    if cfg.INPUT.DATASET_NAME == "SpectrumNet":
        dataset = SpectrumNet(cfg=cfg, Logger=Logger)
    else:
        raise NotImplementedError(f"{cfg.INPUT.DATASET_NAME} is not supported for now.")

    train_set = dataset.train
    test_set = dataset.test

    train_set = preprocessor(subset=train_set, cfg=cfg, Logger=Logger, is_train=True)
    test_set_dict = {}

    for geo_type in cfg.INPUT.HYBRID_GEO:
        cur_testset = preprocessor(subset=test_set[geo_type], cfg=cfg, Logger=Logger, is_train=False)
        test_set_dict[geo_type] = cur_testset

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=min(cfg.SOLVER.BATCH_SIZE, len(train_set)),
        shuffle=True,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        collate_fn=collate_fn
    )

    test_loader_dict = {}

    for geo_type in cfg.INPUT.HYBRID_GEO:
        test_loader_dict[geo_type] = DataLoader(
            dataset=test_set_dict[geo_type],
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            # num_workers=0 if cfg.DEBUG.OVERFIT else cfg.SOLVER.NUM_WORKERS,
            collate_fn=collate_fn
        )

    Logger.info(f"TrainSet and TestSet has been successfully loaded.")

    return train_loader, test_loader_dict



if __name__ == "__main__":
    import argparse
    from configs import cfg
    parser = argparse.ArgumentParser(description="RME")
    parser.add_argument("--config_file", default="./c_configs/SpectrumNet.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_loader, test_loader = make_dataloader(cfg=cfg)

    for batch in train_loader:
        radiomap, sampled_map, building, terrain, f, env = batch
        print((np.argwhere(building != 0.)).shape)
        print((np.argwhere(sampled_map != 255.)).shape)
  
        #print(f)
        #print(env)
        exit()

# python -m c_datasets.loaders
    


