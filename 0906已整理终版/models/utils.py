import torch
import math
import warnings

def value_embedding(heightsteps, dim, max_period=3000):
    """
    Create sinusoidal timestep embeddings.
    :param heightsteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.

    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=heightsteps.device)
    args = heightsteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding

def height_embed(height_idx, cur_batch_size, device, cfg, is_test=False):
    """
    ### Functions:
    - Embed the heights into a particular height map,
        with each height corresponds to a specific height map

    ### Args:
    - `height_idx`: is in [0, 1, 2]

    ### Returns:
    - height_map, .shape = (B, 1, 128, 128)
    """
    height_list = torch.tensor(cfg.CONDITION.HEIGHT_LIST)
    height = height_list[height_idx]

    vec = value_embedding(
        heightsteps=height.view(-1),
        dim=512,
        max_period=5000
    )

    #vec = vec.repeat(cfg.SOLVER.BATCH_SIZE, 1) #BUG
    vec = vec.repeat(cur_batch_size, 1)
    if is_test:
        assert cur_batch_size == 1 # (1, 512)
    vec = vec.to(device)

    if not is_test:
        vec = vec.repeat(1, 32) # 32 * 512 = 128 * 128 # (B, 512 * 32)
        #vec = vec.view(cfg.SOLVER.BATCH_SIZE, 1, 128, 128).to(device) # BUG
        vec = vec.view(cur_batch_size, 1, 128, 128).to(device)
    elif is_test:
        vec = vec.repeat(cfg.TEST.RERANK_FORCE, 32) # (N, 512 * 32)
        vec = vec.view(cfg.TEST.RERANK_FORCE, 1, 128, 128) # (N, 1, 128, 128)

    return vec


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

    dummy = 1
    output = height_embed(
        height_idx=dummy,
        cfg=cfg
    )

    print(output.shape) # (B, 512)
# python -m c_models.utils

