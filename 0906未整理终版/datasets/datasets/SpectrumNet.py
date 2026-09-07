
import sys
import os.path as osp
import os
from torch.utils.data import Dataset, DistributedSampler
import glob
from collections import defaultdict

current_dir = osp.dirname(osp.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from datasets.utils import BaseDataset
from tqdm import tqdm

class SpectrumNet(BaseDataset):
    def __init__(self, cfg, Logger):
        super().__init__(cfg=cfg, Logger=Logger)

        self.train = []
        self.test = []

        self.train, self.test = self.prepare_train_test(cfg=cfg)
        self.presentation(trainset=self.train, testset=self.test, cfg=cfg)

    def prepare_train_test(self, cfg):
        to_be_mixed = cfg.INPUT.HYBRID_GEO

        INVERTED_GEO_TYPES = {v: k for k, v in (cfg.INPUT.GEO_TYPES).items()}

        meta_info = defaultdict(list)

        terrains_txts = sorted(glob.glob(osp.join(self.info_dir, "*")))

        for terrain_txt in terrains_txts:
            terr_idx = (osp.basename(terrain_txt).split("."))[0]
            terr_idx = str(terr_idx)

            with open(terrain_txt, "r") as cur_terr:
                meta_info[terr_idx] = [line.rstrip("\n") for line in cur_terr.readlines()]

        train_info = []
        test_info = {}

        train_data = []
        test_data = []

        # print(f"Received to_be_mixed: {to_be_mixed}.")
        if len(to_be_mixed) >= 1:
            freq_id = cfg.CONDITION.TARGET_FREQUENCY

            for geo in to_be_mixed:
                assert geo in (cfg.INPUT.GEO_TYPES).values(), f"Received geo type {geo}, not a suitable type."

                geo_idx = str(INVERTED_GEO_TYPES[geo])

                geo_samples = meta_info[geo_idx]

                if freq_id.lower() != "all":
                    flag = f"_{freq_id}_"
                    geo_samples = [p for p in geo_samples if flag in osp.basename(p)]

                # geo_txt = f"{geo_idx}.{to_be_mixed[0]}_info.txt"
                # geo_path = osp.join(self.info_dir, geo_txt)

                curr_size = len(geo_samples)
                assert curr_size > 0, f"Plz check the status of geo_samples, got len(geo_samp)=0."

                assert cfg.INPUT.TRAIN_PERCENT > 0.5
                boarderline = int(curr_size * (cfg.INPUT.TRAIN_PERCENT - 0.5))
                train_info += geo_samples[: : 2]
                train_info += geo_samples[1: : 2][: boarderline]

                # BUG [Fixed on 0904] Test set shouldn't be mixed
                # BUG test_info += geo_samples[1: : 2][boarderline: ]   
                test_info[geo] = geo_samples[1: : 2][boarderline: ] 
        else:
            raise ValueError(f"Got target geo type {to_be_mixed}, which is wrong.")

        self.logger.info(f"Meta info for {to_be_mixed} has been collected.")

        # BUG [Fixed on 0903] Need to dynamically sample the radiomaps.
        if False:
            for img_path_z00 in tqdm(train_info, desc=f"Processing TrainSet"):
                radiomap, building, terrain, frequency, envrionment = self.extract_single_sample(img_path=img_path_z00)
                sampled_map = self.sample_all_heights_coordinate(
                    radiomap=radiomap, building=building, num_sample=cfg.INPUT.NUM_SAMPLE,
                    separately=cfg.INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY
                )
                train_data.append((radiomap, sampled_map, building, terrain, frequency, envrionment))

            for img_path_z00 in tqdm(test_info, desc=f"Processing TestSet"):
                radiomap, building, terrain, frequency, envrionment = self.extract_single_sample(img_path=img_path_z00)
                sampled_map = self.sample_all_heights_coordinate(
                    radiomap=radiomap, building=building, num_sample=cfg.INPUT.NUM_SAMPLE,
                    separately=cfg.INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY
                )
                test_data.append((radiomap, sampled_map, building, terrain, frequency, envrionment))

        # return train_data, test_data
        return train_info, test_info

    
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

    dataset = SpectrumNet(cfg=cfg)
    train_set = dataset.train
    test_set = dataset.test

    rm, sm, bd, tr, f, env = train_set[0]

    print(f"{bd.shape}")
    exit()


# python -m c_datasets.datasets.SpectrumNet
