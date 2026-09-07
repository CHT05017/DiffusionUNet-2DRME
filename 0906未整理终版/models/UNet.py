import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)



class PositionalEncoding(nn.Module):
    """
    ### Function:
    - Time step encoding. 
    
    ### `init()` Args:
    - `max_step`: Total number of steps during diffusion.
    - `d_model`: Embedding dimension.
    
    ### `forward()` Args:
    - `t`: Time step. `t <= max_step`
    """
    def __init__(self, max_step: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_step, d_model)
        i_seq = torch.linspace(0, max_step - 1, max_step)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_step, d_model)

        self.embedding = nn.Embedding(max_step, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False) 

    def forward(self, t):
        return self.embedding(t)


class ResBlock(nn.Module):
    """
    ### Function:
    - Pre-Activated kind of ResBlock
    ### `init()` Args:
    - `shape`: (C, H, W)
    """
    def __init__(self,
                 shape,
                 out_channel, 
                 time_channel, 
                 cfg=None
                 ):
        super().__init__()
        in_channel = shape[0]

        assert shape[0] % 32 == 0 and out_channel % 32 == 0, \
            f"Got shape[0]={shape[0]}, out_channel={out_channel}, which don't satisfy x%32==0."
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=shape[0])
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channel)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

        self.time_layer = nn.Linear(time_channel, out_channel)

        self.act = nn.SiLU()

        if in_channel == out_channel:
            self.res_conv = nn.Identity()
        else:
            self.res_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time):
        B = time.shape[0]

        out = self.conv1(self.act(self.norm1(x)))

        t = self.time_layer(self.act(time)) # (B, out_c)
        t = t.reshape(B, -1, 1, 1) # (B, out_c, 1, 1)
        out = out + t

        out = self.conv2(self.act(self.norm2(out)))
        residual = self.res_conv(x)

        out = out + residual
        return out


class SelfAttention(nn.Module):
    """
    2-D Self Attention
    """
    def __init__(self, 
                 shape,
                 dim, 
                 cfg=None):
        super().__init__()
        assert shape[0] % 32 == 0
        self.norm = nn.GroupNorm(num_groups=32, num_channels=shape[0])

        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        norm_x = self.norm(x)

        Q = self.q(norm_x) # (B, C, H, W)
        K = self.k(norm_x)
        V = self.v(norm_x)

        Q = Q.reshape(B, C, H * W)
        Q = Q.permute(0, 2, 1) # (B, H*W, C)

        K = K.reshape(B, C, H * W)
        QK = torch.bmm(Q, K) / (C ** 0.5)
        QK = torch.softmax(QK, dim=-1)

        QK = QK.permute(0, 2, 1)

        V = V.reshape(B, C, H * W)
        res = torch.bmm(V, QK)
        res = res.reshape(B, C, H, W)
        res = self.proj(res)

        return x + res


class ResAttnBlock(nn.Module):
    """
    ResAttn (Combination of ResBlock and SA [Optional])
    ### `init()` Args:
    - `self_attn`: Whether to include the self attention block
    - `double_res`: Whether to include the second ResBlock after `attn_block`.
    """
    def __init__(self, 
                 shape,
                 time_channel,
                 out_channel,
                 self_attn=False,
                 double_res=False,
                 cfg=None):
        super().__init__()

        self.res_block = ResBlock(
            shape=shape,
            time_channel=time_channel,
            out_channel=out_channel,
            cfg=cfg
        )
        self.attn_block = SelfAttention(
            shape=(out_channel, shape[1], shape[2]),
            dim=out_channel,
            cfg=cfg
        )
        self.scd_res_block = ResBlock(
            shape=(out_channel, shape[1], shape[2]),
            out_channel=out_channel,
            time_channel=time_channel,
            cfg=cfg
        )

        if not self_attn:
            self.attn_block.requires_grad_(False)

        if not double_res:
            self.scd_res_block.requires_grad_(False)

        self.attn = self_attn
        self.second_res = double_res

    def forward(self, x, time):
        out = self.res_block(x, time)
        if self.attn:
            out = self.attn_block(out)
        if self.second_res:
            out = self.scd_res_block(out, time)
        return out


class UNet(nn.Module):
    """
    ### `init()` Args:
    - `input_shape`: (C, H, W)
    """
    def __init__(self, 
                 input_shape,
                 cfg
                 ):
        super().__init__()

        C, H, W = input_shape
        channels = cfg.MODEL.LAYER_CHANNELS
        use_sa = cfg.MODEL.USE_SA
        double_res = cfg.MODEL.DOUBLE_RES

        assert len(use_sa) == len(channels) == len(double_res), \
            f"Got use_sa={use_sa}, channels={channels}, double_res={double_res}, not equal in length."
        
        self.conv_in = nn.Conv2d(C, channels[0], 3, 1, 1)

        resolutions = []
        cur_H, cur_W = H, W
        for _ in range(len(channels)):
            resolutions.append((cur_H, cur_W))
            cur_H, cur_W = cur_H // 2, cur_W // 2

        self.time_pe = PositionalEncoding(max_step=cfg.MODEL.DIFFUSION_STEPS, 
                                          d_model=cfg.MODEL.GLOBAL_TIME_DIM)

        time_hidden = 4 * channels[0]
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.MODEL.GLOBAL_TIME_DIM, time_hidden),
            nn.SiLU(),
            nn.Linear(time_hidden, time_hidden)
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.down_proj = nn.ModuleList()
        self.up_proj = nn.ModuleList()

        prev_channel = channels[0]
        for channel, resolution, self_attn, two_res in zip(channels[:-1], resolutions[:-1], 
                                                  use_sa[:-1], double_res[:-1]):
            current_layer = nn.ModuleList([
                ResAttnBlock(
                    shape=(prev_channel if idx % cfg.MODEL.LAYER_BLOCK_NUM == 0 else channel, 
                           resolution[0], resolution[1]),
                    time_channel=time_hidden,
                    out_channel=channel,
                    self_attn=self_attn,
                    double_res=two_res,
                    cfg=cfg
                )
            for idx in range(cfg.MODEL.LAYER_BLOCK_NUM)])

            self.encoders.append(current_layer)
            self.down_proj.append(
                nn.Conv2d(channel, channel, 2, 2)
            )
            prev_channel = channel

        channel = channels[-1]
        last_H, last_W = resolutions[-1]
        self.bottleneck_blk = ResAttnBlock(
            shape=(prev_channel, last_H, last_W),
            time_channel=time_hidden,
            out_channel=channel,
            self_attn=use_sa[-1],
            double_res=double_res[-1],
            cfg=cfg
        )

        prev_channel = channel
        for channel, resolution, self_attn, two_res in zip(
            channels[-2::-1], resolutions[-2::-1], use_sa[-2::-1], double_res[-2::-1]
        ):
            self.up_proj.append(
                nn.ConvTranspose2d(prev_channel, channel, 2, 2)
            )

            current_layer = nn.ModuleList([
                ResAttnBlock(
                    shape=(2 * channel, 
                           resolution[0],
                           resolution[1]
                        ),
                    time_channel=time_hidden,
                    out_channel=channel,
                    self_attn=self_attn,
                    double_res=two_res,
                    cfg=cfg
                )
            for idx in range(cfg.MODEL.LAYER_BLOCK_NUM)])

            self.decoders.append(current_layer)
            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, 1, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x, time): # time.shape: (batch_size,)
        time = self.time_pe(time)
        time_embed = self.time_proj(time)

        x = self.conv_in(x)

        encoders_outs = []
        for encoder, down in zip(self.encoders, self.down_proj):
            blk_outs = []
            for blk in encoder:
                blk: ResAttnBlock
                x = blk(x, time_embed)
                blk_outs.append(x)

            blk_outs = list(reversed(blk_outs)) # FIFO stack
            encoders_outs.append(blk_outs)
            x = down(x)

        x = self.bottleneck_blk(x, time_embed)

        for decoder, up, encoder_out in zip(
            self.decoders, self.up_proj, reversed(encoders_outs)
        ):
            x = up(x)

            for blk_idx, blk in enumerate(decoder):
                x = torch.cat((encoder_out[blk_idx], x), dim=1)
                x = blk(x, time_embed)

        x = self.conv_out(self.act(x))
        return x

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

    dummy = torch.randn(size=(3, 128, 128))

    model = UNet(
        input_shape=(dummy.shape[0], dummy.shape[1], dummy.shape[2]),
        cfg=cfg
    )

    B = 8
    C = dummy.shape[0]
    H = dummy.shape[1]
    W = dummy.shape[2]
    batch_dummy = torch.randn(size=(B, C, H, W))
    batch_time = torch.randint(
        low=0,
        high=cfg.MODEL.DIFFUSION_STEPS,
        size=(B,),
        dtype=torch.long,
        device=batch_dummy.device,
    )
    output = model(batch_dummy, batch_time)
    print(output)
    print(output.shape) # (B, C=1, H, W)
    

    import matplotlib.pyplot as plt
    import numpy as np
    idx = np.random.choice(output.shape[0], size=1, replace=False)
    data = output.detach().numpy()
    data = (data[idx]).squeeze()
    plt.figure()
    plt.imshow(data, cmap="hot", interpolation="bilinear")
    plt.show()

# python -m c_models.UNet



        
            
        




        

        

