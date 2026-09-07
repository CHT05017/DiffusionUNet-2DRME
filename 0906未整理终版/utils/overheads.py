import torch
from thop import profile, clever_format

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return total_params, trainable_params

def macs_flops(model, B=1, C=3, H=128, W=128, device=None, cfg=None):

    model.eval()
    device = next(model.parameters()).device

    x = torch.randn((B, C, H, W), device=device)
    t = torch.randint(
        low=0,
        high=cfg.MODEL.DIFFUSION_STEPS,
        size=(1,),
        dtype=torch.long,
        device=device,
    )

    macs, params = profile(
        model,
        inputs=(x, t),
        verbose=False,
    )

    flops = 2 * macs

    macs, flops, params = clever_format(
        [macs, flops, params],
        "%.3f",
    )

    return macs, flops