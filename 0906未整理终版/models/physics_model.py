
import torch
import numpy as np
import os
from scipy.interpolate import Rbf
from sklearn.cluster import mean_shift
from functools import partial
from scipy.optimize import least_squares

DELTA_HEIGHT_LIST = [0, 2.85, 19.85]
def func(RSS,d,i):
    return np.asarray([RSS_v-np.sum(i/(d_v**2)) for RSS_v, d_v in zip(RSS,d)])

def rbf_tx_surface(point_cloud, building_0, target_height):
    """
    ### Args:
    - point_cloud: (N, 4), [row, col, height, RSS], 
        extracted directly from sampled_map
    - building_0:  (128, 128)

    ### Returns:
    - interpolated_map: (128, 128)
    """
    x_grid, y_grid = np.mgrid[0:128, 0:128]
    z_grid = np.full(shape=(128, 128), fill_value=target_height, dtype=np.float32)

    interpolator = Rbf(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        point_cloud[:, 3],
        function="gaussian",
        smooth=1e-3
    )

    interpolated_map = interpolator(x_grid, y_grid, z_grid)
    interpolated_map = np.clip(interpolated_map, -1, 1)
    interpolated_map[building_0 >= 1] = -1

    return interpolated_map


def get_transmitter_pos(interpolated_map):
    """
    ### Args:
    - interpolated_map: (128, 128)

    ### Returns:
    - centers: (num_tx, 2)
    """
    flat_arr = interpolated_map.flatten()
    top_indices = np.argpartition(-flat_arr, 100)[:100]

    high_rss_points = np.column_stack(
        np.unravel_index(top_indices, interpolated_map.shape
        ))

    centers, _ = mean_shift(high_rss_points, bandwidth=10)
    if len(centers) == 0:
        max_position = np.unravel_index(np.argmax(interpolated_map), interpolated_map.shape)
        centers = np.asarray([max_position], dtype=np.float32)

    if len(centers) > 2:
        center_scores = []

        for center in centers:
            row = int(np.clip(np.rint(center[0]), 0, 127))
            col = int(np.clip(np.rint(center[1]), 0, 127))
            center_scores.append(interpolated_map[row, col])

        selected_indices = np.argsort(center_scores)[-2:]
        centers = centers[selected_indices]

    return centers


def estimate_tx_map(
        sampled_map,
        sampled_mask,
        building,
        target_height_idx,
        cfg
):
    """
    ### Functions:
    - to estimate Tx map

    ### Args:
    - sampled_map:  (B, 3, 128, 128)
    - sampled_mask: (B, 3, 128, 128)
    - building:     (B, 3, 128, 128)
    - target_height_idx: 0 / 1 / 2

    ### Returns:
    - tx_map: (B, 1, 128, 128)
    """
    batch_size = sampled_map.shape[0]

    height_list = torch.tensor(cfg.CONDITION.HEIGHT_LIST, dtype=sampled_map.dtype)
    target_height = float(cfg.CONDITION.HEIGHT_LIST[target_height_idx])

    tx_maps = []

    for batch_idx in range(batch_size):
        coordinates = torch.nonzero(sampled_mask[batch_idx] > 0, as_tuple=False)

        sample_height_idx = coordinates[:, 0]
        sample_row = coordinates[:, 1]
        sample_col = coordinates[:, 2]

        sample_height = height_list[sample_height_idx]

        sample_rss = sampled_map[batch_idx, sample_height_idx, sample_row, sample_col]

        point_cloud = torch.stack(
            tensors=(sample_row.to(torch.float32), sample_col.to(torch.float32), sample_height, sample_rss),
            dim=1
        )

        point_cloud = point_cloud.cpu().numpy()
        target_building = building[batch_idx, target_height_idx].cpu().numpy()

        interpolated_map = rbf_tx_surface(point_cloud=point_cloud, building_0=target_building, target_height=target_height)
        transmitter_positions = get_transmitter_pos(interpolated_map=interpolated_map)

        tx_map = torch.zeros(size=(1, 128, 128), dtype=torch.float32)

        for position in transmitter_positions:
            row = int(np.rint(position[0]))
            col = int(np.rint(position[1]))
            tx_map[0, row, col] = 1.0

        tx_maps.append(tx_map)

    return torch.stack(tx_maps, dim=0)


def rbf(point_cloud, building_0):
    x_grid, y_grid = np.mgrid[0:128, 0:128]

    rbf = Rbf(point_cloud[:,0], point_cloud[:,1], point_cloud[:,3], function='gaussian', smooth=1e-3)
    interpolated_values = rbf(x_grid, y_grid)

    interpolated_values_clipped = np.clip(interpolated_values, -1, 1)
    interpolated_values_clipped[building_0 >= 1] = -1

    return interpolated_values_clipped

def pos2radiomap(transmitter_pos, p_1, building, PLE, size=128, LogPathLoss = False):
    B = 3
    S = transmitter_pos.shape[0]  
    m, n = size, size
    noise_std = 0.05

    transmitter_pos = torch.tensor(transmitter_pos)

    x = torch.arange(m).view( -1, 1).repeat(S, 1, n)
    y = torch.arange(n).view( 1, -1).repeat(S, m, 1)


    pos_x = transmitter_pos[:,0].view(S, 1, 1)
    pos_y = transmitter_pos[:,1].view(S, 1, 1)

    Distance_2D = (x - pos_x) ** 2 + (y - pos_y) ** 2

    H_2 = (torch.tensor(DELTA_HEIGHT_LIST)**2).view(B, 1, 1, 1)

    Distance_3D = torch.clamp(torch.sqrt(Distance_2D.view(1, S, m ,n).repeat(B, 1, 1, 1) + H_2), min=1)
    p_1 = torch.tensor(p_1).view(1,S,1,1).repeat(B, 1, m, n)

    if LogPathLoss:
        radio_map_raw = torch.log10(p_1) - PLE * torch.log10(Distance_3D) + noise_std * torch.randn(Distance_3D.size())
        radio_map_db = torch.sum(radio_map_raw, dim=1).clamp(-1,1)
    else:
        radio_map_raw = p_1 / Distance_3D ** PLE 
        radio_map_db = torch.log10(torch.sum(radio_map_raw, dim=1)).clamp(-1,1)
        radio_map_db = (radio_map_db + noise_std * torch.randn(radio_map_db.size())).clamp(-1,1)

    radio_map_db[building >= 1] = -1

    return radio_map_db

def get_transmitter_info(transmitter_pos, point_cloud):
    Pr_dBm = point_cloud[:,3]
    Pr_W = 10**(Pr_dBm)

    d = []
    for pos in transmitter_pos:
        d_i = np.linalg.norm(point_cloud[:,:3] - np.concatenate([pos,[0.15]]), axis=1)
        d.append(d_i)

    func_p = partial(func, Pr_W, np.stack(d,axis=1))

    init_val = np.ones(transmitter_pos.shape[0])*10
    try:
        root = least_squares(func_p, init_val, bounds=(0,np.inf)).x
    except Exception as e:
        print("!!!least_squares ERROR, use default value!!!")
        print(e)
        root = np.ones(transmitter_pos.shape[0])*10
    return 2, root

def get_phy_radiomap(point_cloud, building, point_cloud_height):

    phy_radiomap_list = []
    for point, b  in zip(point_cloud, building):
        radiomap_CZ = rbf(point.cpu(), b[point_cloud_height].cpu())
        pos = get_transmitter_pos(radiomap_CZ)
        n, p_1 = get_transmitter_info(pos, point.cpu())
        phy_radiomap = pos2radiomap(pos, p_1, b.cpu(), n)
        if torch.any(torch.isnan(phy_radiomap)):
            print("!!!There is NAN here, check the data!!!")
            print(p_1,pos)
        phy_radiomap_list.append(phy_radiomap)

    return torch.stack(phy_radiomap_list).to(torch.float32) #  ,pos

def RSS_project(sampled_map, sampled_mask, building, env, target_height_idx, cfg, pointcloud):
    """
    ### Function:
    - Convert sampled RSS values from other height layers to the target height layer,
        which is essentially sort of data augmentation to overcome extrmely low sampling rate.

    ### Args:
    - `sampled_map`: (B, C, H, W)
    - `building`: (B, C, H, W)
    - `env`: (B, )
    """
    if cfg.DEBUG.SINGLE_HEIGHT_BASELINE:
        expected_height = int(cfg.CONDITION.TARGET_HEIGHT)

        if target_height_idx != expected_height:
            raise ValueError()

        observed = sampled_map[:, target_height_idx:target_height_idx + 1].clone()
        observed_mask = sampled_mask[:, target_height_idx:target_height_idx + 1].bool()

        free_space = (building[:, target_height_idx:target_height_idx + 1] == 0)
        observed_mask = observed_mask & free_space
        observed = observed.masked_fill(~observed_mask, -1.0)

        return observed.float(), observed_mask.float()    

    canvas = sampled_map[:, target_height_idx: target_height_idx + 1].clone()
    canvas_mask = sampled_mask[:, target_height_idx: target_height_idx + 1].bool()

    category = cfg.INPUT.GEO_CAT_ONE
    proj_bias = cfg.PHYSICS.HATA.PROJ_BIAS
    lower_bound = cfg.PHYSICS.HATA.LOWER_BOUND
    heights = cfg.CONDITION.HEIGHT_LIST

    GEO = {int(k): v for k, v in category.items()}
    batch = sampled_map.shape[0]

    for layer_idx in range(len(heights)):
        if layer_idx == target_height_idx:
            continue

        source = sampled_map[:, layer_idx: layer_idx + 1]
        source_mask = sampled_mask[:, layer_idx: layer_idx + 1].bool()
        # source_valid = source > lower_bound[layer_idx]
        source_valid = source_mask & (source > lower_bound[layer_idx])

        # target_empty = canvas == -1
        target_empty = ~canvas_mask

        proj_mask = source_valid & target_empty # (B, 1, 128, 128)

        # BUG [Fixed] Not suitable for mix-env training
        # projected = source + biases[layer_idx][target_height_idx]
        batch_bias = []
        for batch_idx in range(batch):
            env_id = int(env[batch_idx].item())
            geo_type = GEO[env_id]

            bias = proj_bias[geo_type][layer_idx][target_height_idx]
            batch_bias.append(bias)

        batch_bias = torch.tensor(batch_bias, dtype=sampled_map.dtype, 
                                  device=sampled_map.device) # (B,)
        batch_bias = batch_bias.view(batch, 1, 1, 1)
        projected = source + batch_bias

        canvas[proj_mask] = projected[proj_mask]
        canvas_mask[proj_mask] = True

    blocked_area = building[:, target_height_idx:target_height_idx + 1] == 1
    canvas[blocked_area] = -1
    canvas_mask[blocked_area] = False

    f = get_phy_radiomap(
        point_cloud=pointcloud,
        building=building,
        point_cloud_height=target_height_idx
    )
    f = f[:, target_height_idx: target_height_idx + 1, :, :]
    f[canvas == -1] = -1

    w = pow(2,-heights[target_height_idx] / 10)
    output = canvas * w + f.to(canvas.device) * (1 - w)

    # return canvas.to(torch.float32), canvas_mask.to(torch.float32), output.to(torch.float32)
    return output.to(torch.float32), canvas_mask.to(torch.float32)

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

    B = 8
    dum_sampled_map = torch.randn((B, 3, 128, 128))
    dum_building = torch.randint_like(input=dum_sampled_map, low=0, high=1)
    dum_env = torch.tensor([2] * B)
    dum_tg_height_idx = 2

    output = RSS_project(sampled_map=dum_sampled_map,
                         building=dum_building,
                         env=dum_env,
                         target_height_idx=dum_tg_height_idx,
                         cfg=cfg)

    print(output.shape) # (B, 1, H, W)

# python -m c_models.physics_model
