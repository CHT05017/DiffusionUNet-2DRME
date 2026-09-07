import glob
from tqdm import tqdm
import os.path as osp
import os
from accelerate import Accelerator

def create_z00_info_txt(cfg, Logger):
    root = cfg.INPUT.DATASET_ROOT

    img_dirs = sorted(
        path for path in glob.glob(osp.join(root, "png", "*"))
        if osp.isdir(path))
    info_dir = osp.join(root, "info")

    if not osp.exists(info_dir):
        os.makedirs(info_dir, exist_ok=True)

    terrain_names = list(map(lambda path: osp.basename(path), img_dirs))
    for terrain in tqdm(terrain_names, desc=f"Extracting z00 imgs"):
        terrain_txt = osp.join(info_dir, f"{terrain}_info.txt")

        imgs = sorted(glob.glob(osp.join(
            root, "png", terrain, "*"
        )))

        with open(terrain_txt, "w") as cur_info:
            for img_path in imgs:
                if "z00" in img_path:
                    cur_info.write(img_path)
                    cur_info.write("\n")

    Logger.info(f"z00 imgs are extracted!")
