import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from model.ddpm_trans_modules.trans_block_eca import TransformerBlock_eca

from model.spatial_attention import SpatialTransformer
import numpy as np

def gaussian_pdf(miu, sigma, x):
    return np.exp(-(x - miu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

def generate_gaussian_mask(units, units_num):
    sigmas = np.linspace(5, 0.5, len(units), dtype=np.float32)
    # print(sigmas)
    x = np.linspace(0, units_num, units_num, dtype=np.float32)
    y = 0
    for i, sigma in enumerate(sigmas):
        sig = math.sqrt(sigma)
        y += gaussian_pdf(units[i], sig, x)
    yt = torch.from_numpy(y)
    yt = yt.view(1, units_num, 1, 1)
    return yt


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                # calculate std and reshape
    return mean, std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def AdaIn(content, style, edit=False):
    assert content.shape[:2] == style.shape[:2] # Only first two dim, such that different image sizes is possible
    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)
    if edit:
        print('before:', mean_style[:, 17, :, :])
        mean_style[:, 17, :, :] =  -5
        print('after:', mean_style[:, 17, :, :])
    output = std_style*((content - mean_content) / (std_content)) + mean_style # Normalise, then modify mean and std
    return output

# model

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBloc_eca(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_eca(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channel=6,
            inner_channel=32,
            norm_groups=32,
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)


        # self.cross_att1 = SpatialTransformer(dim, 1, dim, context_dim=768)

        self.cross_att2 = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att3 = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2, context_dim=768)
        self.cross_att4 = SpatialTransformer(dim * 2 ** 3, 1, dim * 2 ** 3, context_dim=768)


        ####### control net #########
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(16, 16, 3, padding=1),
            # nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(32, 32, 3, padding=1),
            zero_module(nn.Conv2d(32, dim, 3, padding=1))
        )
        self.conv2_control = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.PixelUnshuffle(2))
        self.conv3_control = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv4_control = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.block1_control = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2_control = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3_control = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4_control = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)

        self.conv_up3 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)

        self.decoder_block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)


        # self.cross_att1_de = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att2_de = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att3_de = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2, context_dim=768)

        ########## style transfer ##########

        self.units1 = [0, 6, 15, 11, 17]
        self.units2 = [50, 15, 45, 70, 17, 85, 55, 24, 49, 84, 60, 48, 89, 74, 80, 52, 78, 83, 94, 63, 58]


        self.mask1 = nn.Parameter(generate_gaussian_mask(self.units1, dim * 2 ** 0), requires_grad=False)
        self.mask2 = nn.Parameter(generate_gaussian_mask(self.units2, dim * 2 ** 1), requires_grad=False)

    def forward(self, x, t, style, context):
        x_style = self.input_hint_block(style)
        x = self.conv1(x)
        x1 = self.block1(x, t)

        x2 = self.conv2(x1)
        x2 = self.block2(x2, t)
        x2 = self.cross_att2(x2, context)

        x3 = self.conv3(x2)
        x3 = self.block3(x3, t)
        x3 = self.cross_att3(x3, context)

        x4 = self.conv4(x3)
        x4 = self.block4(x4, t)
        x4 = self.cross_att4(x4, context)


        ######## control net forward ##########

        x_context1 = self.block1_control(x_style, t)
        tmp_1 = x_context1 * torch.cat([self.mask1] * x_context1.shape[0], dim=0)
        x_context1 = x_context1 + tmp_1

        x_context2 = self.conv2_control(x_context1)
        x_context2 = self.block2_control(x_context2, t)

        tmp_2 = x_context2 * torch.cat([self.mask2] * x_context2.shape[0], dim=0)
        x_context2 = x_context2 + tmp_2


        x_context3 = self.conv3_control(x_context2)
        x_context3 = self.block3_control(x_context3, t)


        x_context4 = self.conv4_control(x_context3)
        x_context4 = self.block4_control(x_context4, t)

        x4 = AdaIn(x4, x_context4)

        de_level3 = self.conv_up3(x4)
        de_level3 = torch.cat([de_level3, x3], 1)
        de_level3 = self.conv_cat3(de_level3)

        de_level3 = AdaIn(de_level3, x_context3)

        de_level3 = self.decoder_block3(de_level3, t)
        de_level3 = self.cross_att3_de(de_level3, context)
        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)

        de_level2 = AdaIn(de_level2, x_context2)

        de_level2 = self.decoder_block2(de_level2, t)
        de_level2 = self.cross_att2_de(de_level2, context)
        de_level1 = self.conv_up1(de_level2)

        de_level1 = AdaIn(de_level1, x_context1)

        de_level1 = torch.cat([de_level1, x1], 1)

        mid_feat = self.decoder_block1(de_level1, t)
        mid_feat = AdaIn(mid_feat, x_context2)

        return mid_feat, de_level2

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = 1e-5

    def forward(self, x, y):
        mean_x, mean_y = torch.mean(x, dim=(2, 3), keepdim=True), torch.mean(y, dim=(2, 3), keepdim=True)
        std_x, std_y = torch.std(x, dim=(2, 3), keepdim=True) + self.eps, torch.std(y, dim=(2, 3), keepdim=True) + self.eps
        return std_y * (x - mean_x) / std_x + mean_y

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        dim = inner_channel

        self.encoder_water = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)


        self.refine = ResnetBloc_eca(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))
        self.de_predict_support = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))


    def forward(self, x, time, style, context):

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        mid_feat, x1 = self.encoder_water(x, t, style, context)


        mid_feat2 = self.refine(mid_feat, t)

        return self.de_predict(mid_feat2)

