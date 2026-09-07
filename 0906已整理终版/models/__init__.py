import torch
from models.UNet import UNet
from utils.overheads import count_params, macs_flops

def make_UNet(
        cfg,
        Logger
):

    UNET_INPUT_CHANNEL = 6
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(
        # [x_t, physics_map, height_map, building, terrain]
        input_shape=(UNET_INPUT_CHANNEL, 128, 128), # SpectrumNet
        cfg=cfg
    )

    model = model.to(device=device)
    total, trainable = count_params(model=model)

    macs, flops = macs_flops(model=model, device=device, cfg=cfg, C=UNET_INPUT_CHANNEL)

    Logger.info(f"UNet has been successfully loaded.")
    Logger.info(f"Model device: {device}")
    Logger.info(f"Model total parameters: {total / 1e6:.2f} M")
    Logger.info(f"Model trainable parameters: {trainable / 1e6:.2f} M")
    Logger.info(f"Model MACs: {macs}; FLOPs: {flops}.")

    return model
    