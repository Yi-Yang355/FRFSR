import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.modules.batchnorm import _BatchNorm

from torch import nn, Tensor
from typing import Optional, Tuple
from mmsr.models.archs.ddf.ddf import DDFPack

# TODO: modify it.
def srntt_init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def default_init_weights(module_list, scale=1):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(basic_block, n_basic_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        n_basic_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(n_basic_blocks):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m.expand_as(x)

class ResidualDDFBlock(nn.Module):
    def __init__(self, nf=64, stride=1, first_dilation=1, se_ratio=0.2):
        super(ResidualDDFBlock, self).__init__()
        
        self.conv1 = DDFPack(nf, kernel_size=3, stride=1,
                             dilation=first_dilation, se_ratio=se_ratio, kernel_combine='mul')
        self.conv2 = DDFPack(nf, kernel_size=3, stride=1,
                                     dilation=first_dilation, se_ratio=se_ratio, kernel_combine='mul')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out
 
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        sn (bool): Whether to use spectral norm. Default: False.
        n_power_iterations (int): Used in spectral norm. Default: 1.
        sn_bias (bool): Whether to apply spectral norm to bias. Default: True.

    """

    def __init__(self,
                 nf=64,
                 res_scale=1,
                 pytorch_init=False,
                 sn=False,
                 n_power_iterations=1,
                 sn_bias=True,
                 use_ESA=True):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.use_ESA = use_ESA
        if sn:
            self.conv1 = spectral_norm(
                self.conv1,
                name='weight',
                n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(
                self.conv2,
                name='weight',
                n_power_iterations=n_power_iterations)
            if sn_bias:
                self.conv1 = spectral_norm(
                    self.conv1,
                    name='bias',
                    n_power_iterations=n_power_iterations)
                self.conv2 = spectral_norm(
                    self.conv2,
                    name='bias',
                    n_power_iterations=n_power_iterations)
        if self.use_ESA:
            self.ESA = ESA(nf)
        self.relu = nn.ReLU(inplace=True)

        if not sn and not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        if self.use_ESA:
            out = self.ESA(out)
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        nf (int): Channel number of intermediate features.
    """

    def __init__(self, scale, nf):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(nf, 4 * nf, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(nf, 9 * nf, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
        


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (N, C, H, W).
        flow (Tensor): Tensor with size (N, H, W, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'The size type should be ratio or shape, but got type {size_type}.'
        )

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: modify it.
def pixel_unshuffle(x, s):
    """ Pixel unshuffle.

    Args:
        x (Tensor): the input feature. The shape is [b, c, hh, hw].
        s (int): downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (s**2)
    assert hh % s == 0 and hw % s == 0
    h = hh // s
    w = hw // s
    x_view = x.view(b, c, h, s, w, s)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


# TODO: modify it.
def tensor_shift(x, shift=(2, 2), fill_val=0):
    """ Tensor shift.

    Args:
        x (Tensor): the input tensor. The shape is [b, h, w, c].
        shift (tuple): shift pixel.
        fill_val (float): fill value

    Returns:
        Tensor: the shifted tensor.
    """

    _, h, w, _ = x.size()
    shift_h, shift_w = shift
    new = torch.ones_like(x) * fill_val

    if shift_h >= 0 and shift_w >= 0:
        len_h = h - shift_h
        len_w = w - shift_w
        new[:, shift_h:shift_h + len_h,
            shift_w:shift_w + len_w, :] = x.narrow(1, 0,
                                                   len_h).narrow(2, 0, len_w)
    else:
        raise NotImplementedError
    return new
