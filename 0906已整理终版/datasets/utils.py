from torch.utils.data import Dataset
import torch
import sys
import os.path as osp
current_dir = osp.dirname(osp.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import re
import cv2
import numpy as np
from tqdm import tqdm

class preprocessor(Dataset):
    """
    ### Init Args:
    - `subset`: train_set or test_set
    """
    def __init__(self, subset, Logger=None, cfg=None, is_train=True):
        super().__init__()
        self.dataset = subset
        self.map_shape = cfg.INPUT.MAP_SHAPE
        self.height_shape = len(cfg.CONDITION.HEIGHT_LIST)
        self.cfg = cfg
        self.base_dataset = BaseDataset(cfg, Logger=Logger)
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idxes):
        return self.__getsingleitem__(idxes)

    def __getsingleitem__(self, index):
        #radiomap, sampled_map, building, terrain, frequency, env = self.dataset[index]
        img_path = self.dataset[index]

        # resample every time trying to grab a batch of data
        radiomap, building, terrain, frequency, env = self.base_dataset.extract_single_sample(img_path=img_path)

        # Diffusion Normalization
        # (0, 1) -> (-1, 1)
        radiomap = radiomap.astype(np.float32) * 2.0 / 255.0 - 1.0
        terrain = (terrain + 2) / 190

        if self.cfg.MODEL.SINGLE_HEIGHT:
            height_idx = int(self.cfg.CONDITION.TARGET_HEIGHT)
            num_sample = self.cfg.INPUT.NUM_SAMPLE

            if self.is_train:
                rng = np.random
            else:
                rng = np.random.default_rng(self.cfg.SOLVER.SEEDS + index)
            valid = np.argwhere(building[height_idx] == 0)

            chosen = rng.choice(len(valid), size=num_sample, replace=False)
            y, x = valid[chosen].T

            sampled_map = np.full_like(radiomap, -1.0, dtype=np.float32)
            sampled_mask = np.zeros_like(radiomap, dtype=np.float32)

            sampled_map[height_idx, y, x] = (radiomap[height_idx, y, x]) # only target_height layer contains RSS
            sampled_mask[height_idx, y, x] = 1.0

            pointcloud = np.stack((y, x, np.full(num_sample,
                        self.cfg.CONDITION.HEIGHT_LIST[height_idx]), radiomap[height_idx, y, x]),
                axis=1).astype(np.float32)
        else:
            sampled_map, sampled_mask = self.base_dataset.sample_all_heights_coordinate(
                radiomap=radiomap,
                building=building,
                num_sample=self.cfg.INPUT.NUM_SAMPLE,
                separately=self.cfg.INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY
            )
        pointcloud = self.base_dataset.sample_all_heights_pointcloud(
            radiomap=radiomap,
            building=building,
            cfg=self.cfg
        )

        radiomap: np.ndarray
        refine = lambda x: x.reshape(self.height_shape, self.map_shape[0], self.map_shape[1])
        radiomap, sampled_map, sampled_mask, building = \
                map(refine, [radiomap, sampled_map, sampled_mask, building])

        terrain = terrain.reshape(1, self.map_shape[0], self.map_shape[1])

        turn = lambda x: torch.tensor(x, dtype=torch.float32)
        sampled_map, sampled_mask, building, terrain, frequency, env, radiomap, pointcloud = \
            map(turn, [sampled_map, sampled_mask, building, terrain, frequency, env, radiomap, pointcloud])

        return radiomap, sampled_map, sampled_mask, building, terrain, frequency, env, pointcloud


def collate_fn(batch):
    radiomaps, sampled_maps, sampled_masks, buildings, terrains, frequencies, envs, pointclouds = zip(*batch) 

    return torch.stack(radiomaps, dim=0), torch.stack(sampled_maps, dim=0), torch.stack(sampled_masks, dim=0), \
            torch.stack(buildings, dim=0), torch.stack(terrains, dim=0),  \
            torch.stack(frequencies, dim=0), torch.stack(envs, dim=0), torch.stack(pointclouds, dim=0)

class BaseDataset(object):
    """
    Base Dataset class for SpectrumNet
    """
    def __init__(self, cfg, Logger):
        self.frequencies = cfg.CONDITION.FREQUENCIES_LIST
        self.png_pattern = r"T(\d+)C(\d+)D(\d+)_n(\d+)_f(\d+)"
        self.root = cfg.INPUT.DATASET_ROOT
        self.info_dir = osp.join(self.root, "info")
        self.map_shape = cfg.INPUT.MAP_SHAPE
        self.logger = Logger
        
    def extract_single_sample(self, img_path: str, cfg=None):
        """
        ### Function:
        - Extract the information of a single sample
        
        ### Args:
        - `img_path`: is extracted from `/path/to/{terrain}_info.txt`
        """
        T, C, D, n, f = re.match(self.png_pattern, osp.basename(img_path)).groups()

        radiomap = np.stack([
            cv2.imread(img_path, 0),
            cv2.imread(img_path.replace('z00', 'z01'), 0),
            cv2.imread(img_path.replace('z00', 'z02'), 0)
            ])

        npz_root = osp.join(self.root, "npz")
        npz_path = f"T{T}C{C}D{D}_n{n}_bdtr.npz"
        npz_path = osp.join(npz_root, npz_path)

        # mat.keys()
        # inBldg_zyx, terrain_yx
        mat = np.load(npz_path)

        # building.shape = (3, 128, 128)
        # e.g., for DenseUrban, pixels with buildings will be filled with 1
        # terrain.shape = (128, 128)
        building, terrain = mat["inBldg_zyx"], mat["terrain_yx"]

        frequency = self.frequencies[int(f)]
        environment = int(T)

        return radiomap, building, terrain, frequency, environment

    def separately_sample_certain_height_pointcloud(self, height_idx, building, num_sample, radiomap, cfg):
        """
        ## Not used when Fast=True, only used when Fast=False, where physical data augmentation models are put into use. 

        ### Function:
        - To **separately** sample **ONE designated height**

        ### Args:
        - `height_idx`: <= 2, ([0.15, 3, 20])
        - `building`: .shape = (3, 128, 128)
        - `num_sample`: number of sampled dots
        - `radiomap`: Radio Map

        ### Returns:
        - sampled_points at denoted height, 
        - .shape = (num_sample, 4)
        """
        assert height_idx < 3, f"Wrong height idx, should <= 2"

        wo_building = 1 - (building[height_idx]).reshape(-1)
        prob = wo_building / np.sum(wo_building)

        selected_points = np.random.choice(a=128*128, size=num_sample, replace=False, p=prob)

        sel_pts_x = selected_points // self.map_shape[0]
        sel_pts_y = selected_points % self.map_shape[1]

        sampled_ss = radiomap[height_idx, sel_pts_x, sel_pts_y] # (num_sample, )
        sampled_height = np.ones_like(sampled_ss) * cfg.CONDITION.HEIGHT_LIST[height_idx] # (sparse_num, )

        sampled_points = np.stack([
            sel_pts_x, sel_pts_y, sampled_height, sampled_ss
        ], axis=1)

        return sampled_points

    def hybridly_sample_certain_all_heights_pointcloud(self, radiomap, building, cfg):
        """
        ## Not used when Fast=True, only used when Fast=False, where physical data augmentation models are put into use. 
        ### Functions
        - Separately sample `cfg.num_sample` pixels, then combine them into 3 * `cfg.num_sample` pixels, and finally extract
            `cfg.num_sample` pixels hybridly

        ### Args:
        - `radiomap`
        - `building`
        """
        point_cloud_list = []

        for point_cloud_height in range(3):
            p = 1 - building[point_cloud_height].reshape(-1)
            p = p / np.sum(p)
        
            random_point = np.random.choice(128 * 128, cfg.INPUT.NUM_SAMPLE, replace=False, p = p)
            random_point_x = random_point // 128
            random_point_y = random_point % 128

            radio_ch = radiomap[point_cloud_height, random_point_x, random_point_y]
            height_ch = np.ones_like(radio_ch) * cfg.CONDITION.HEIGHT_LIST[point_cloud_height]

            point_cloud = np.stack([random_point_x, random_point_y, height_ch, radio_ch], axis=1) # (sparse_num, 4)
            point_cloud = torch.tensor(point_cloud).to(torch.float32)

            point_cloud_list.append(point_cloud)
        
        point_cloud_pool = torch.cat(point_cloud_list,dim=0) # (3 * sparse_num, 4)

        point_cloud_list.append(
            point_cloud_pool[np.random.choice(point_cloud_pool.shape[0], cfg.INPUT.NUM_SAMPLE, replace=False)]
            )
        
        return point_cloud_list

    def sample_all_heights_coordinate(self, radiomap, building, num_sample, separately=True):
        """
        ## Always in use no matter Fast=True or Fast=False
        
        ### Function:
        - To sample **all** heights
        - When **separately** is True, `num_sample` pixels are sampled on **EACH** height
        - When **separately** is False `num_sample` pixels are sampled from the entire 3D space

        ### Args:
        - `radiomap`: Radio Map, shape = (3, 128, 128), uint8
        - `building`: Building, shape = (3, 128, 128)
        - `num_sample`: Pixels to be sampled.
        - `separately`: Refer to *Function*.

        ### Returns:
        - sampled_map, .shape = (3, 128, 128)
        - for the sampled pixels, value != -1.0
        """
        
        # sampled_map = np.full_like(radiomap, fill_value = -1.0)
        # normed_radiomap = radiomap.astype(np.float32) * 2.0 / 255.0 - 1.0
        normed_radiomap = radiomap.astype(np.float32)
        sampled_map = np.full(shape=radiomap.shape, fill_value=-1.0, dtype=np.float32)

        sampled_mask = np.zeros_like(radiomap, dtype=np.float32)

        if separately:
            for height in range(3):
                wo_building = np.argwhere(building[height] == 0)
                num_available = len(wo_building)

                assert num_available >= num_sample, \
                    f"Got {num_available} available pixels, but {num_sample} are required."

                sel_idxes = np.random.choice(num_available, size=num_sample, replace=False)

                for sel_idx in sel_idxes:
                    y, x = wo_building[sel_idx]
                    sampled_map[height, y, x] = normed_radiomap[height, y, x]
                    sampled_mask[height, y, x] = 1.0
        else:
            wo_building = np.argwhere(building == 0) # (N, 3)
            num_available = len(wo_building)

            assert num_available >= num_sample, \
                f"Got {num_available} available pixels, but {num_sample} are required."

            sel_idxes = np.random.choice(num_available, size=num_sample, replace=False)
            sel_coords = wo_building[sel_idxes]
            sel_coords = sel_coords.T # (3, num_sample)

            sel_z, sel_y, sel_x = sel_coords[0], sel_coords[1], sel_coords[2]

            sampled_map[sel_z, sel_y, sel_x] = normed_radiomap[sel_z, sel_y, sel_x]
            sampled_mask[sel_z, sel_y, sel_x] = 1.0

        return sampled_map, sampled_mask

    def sample_all_heights_pointcloud(self, radiomap, building, cfg):
        if cfg.INPUT.SAMPLE_EACH_HEIGHT_SEPARATELY:
            pointcloud = self.separately_sample_certain_height_pointcloud(
                height_idx=cfg.CONDITION.TARGET_HEIGHT,
                radiomap=radiomap,
                building=building,
                num_sample=cfg.INPUT.NUM_SAMPLE,
                cfg=cfg
            )
        else:
            pointcloud = self.hybridly_sample_certain_all_heights_pointcloud(
                radiomap=radiomap,
                building=building,
                cfg=cfg
            )
            pointcloud = pointcloud[-1]

        return pointcloud
    


    def presentation(self, trainset: list, testset: dict, cfg):
        line = "=" * 40
        quad = "    "
        subline = "-" * 40
        self.logger.info(line)
        self.logger.info(f"{quad}Dataset Info:")
        if len(cfg.INPUT.HYBRID_GEO) > 1:
            self.logger.info(f"{quad}Mixed Datasets: {cfg.INPUT.HYBRID_GEO}.")
        elif len(cfg.INPUT.HYBRID_GEO) == 1:
            self.logger.info(f"{quad}Got Dataset: {cfg.INPUT.HYBRID_GEO}.")
        self.logger.info(subline)
        self.logger.info(f"{quad}Train Set Info:")
        self.logger.info(f"{quad}Total: {len(trainset)} voxels.")
        self.logger.info(subline)
        self.logger.info(f"{quad}Test Set Info:")
        self.logger.info(f"{quad}Tested Geos: {list(testset.keys())}.")
        for k, v in testset.items():
            self.logger.info(f"{quad}{k}: {len(v)} voxels.")
        self.logger.info(line)


        

        


        





