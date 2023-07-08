import mmsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsr.models.archs.DCNv2.dcn_v2 import DCN_sep_pre_multi_offset as DynAgg


class SwinUnetv3RestorationNet(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(SwinUnetv3RestorationNet, self).__init__()
        self.dyn_agg_restore = DynamicAggregationRestoration(ngf=ngf, n_blocks=n_blocks, groups=groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.bias.data.zero_()


    def forward(self, x, sife, pre_offset_flow_sim, img_ref_feat):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)

        upscale_restore = self.dyn_agg_restore(base, sife, pre_offset_flow_sim, img_ref_feat)

        return upscale_restore + base
    
class DynamicAggregationRestoration(nn.Module):

    def __init__(self,
                 ngf=64,
                 n_blocks=16,
                 groups=8,
                 ):
        super(DynamicAggregationRestoration, self).__init__()

        self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)

        # ---------------------- Down ----------------------

        # dynamic aggregation module for relu1_1 reference feature
        self.down_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
        self.down_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.down_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups, extra_offset_mask=True)

        self.ddf_block_1 = arch_util.make_layer(arch_util.ResidualDDFBlock, 2, nf=ngf)

        # for large scale
        self.down_head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.down_body_large = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.down_tail_large = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)

        # dynamic aggregation module for relu2_1 reference feature
        self.down_medium_offset_conv1 = nn.Conv2d(
            ngf + 128*2, 128, 3, 1, 1, bias=True)
        self.down_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.down_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1,dilation=1,
                                    deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.down_head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
            
        self.ddf_block_2 = arch_util.make_layer(arch_util.ResidualDDFBlock, 2, nf=ngf)
        self.down_body_medium = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)

                
        self.down_tail_medium = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)


        # ---------------------- Up ----------------------
        # dynamic aggregation module for relu3_1 reference feature
        self.up_small_offset_conv1 = nn.Conv2d(
            ngf + 256*2, 256, 3, 1, 1, bias=True)  # concat for diff
        self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)


        # for small scale restoration
        self.up_head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        
        self.ddf_block_3 = arch_util.make_layer(arch_util.ResidualDDFBlock, 2, nf=ngf)
        self.up_body_small = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        
        
        self.up_tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
        # self.up_tail_small = DDFUpPack(in_channels=64, scale_factor=2, joint_channels=64, kernel_combine='mul')
      
        # dynamic aggregation module for relu2_1 reference feature
        self.up_medium_offset_conv1 = nn.Conv2d(
            ngf + 128*2, 128, 3, 1, 1, bias=True)
        self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.up_head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
            
        self.ddf_block_4 = arch_util.make_layer(arch_util.ResidualDDFBlock, 2, nf=ngf)
        self.up_body_medium = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        
        self.up_tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))
        # self.up_tail_medium = DDFUpPack(in_channels=64, scale_factor=2, joint_channels=64, kernel_combine='mul')

        # dynamic aggregation module for relu1_1 reference feature
        self.up_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
        self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)
                                
        self.ddf_block_5 = arch_util.make_layer(arch_util.ResidualDDFBlock, 2, nf=ngf)
                                
        # for large scale
        self.up_head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.up_body_large = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        
        self.up_tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))
        self.high_q_conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
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
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output
    
    def get_flat_mask(self, fea, std_thresh=0.04):
        B, _, H, W = fea.size()
        std_map = torch.std(fea, dim=1, keepdim=True).view(B, 1, H, W)
        mask = torch.lt(std_map, std_thresh).float()

        return mask

    def forward(self, base, sife, pre_offset_flow_sim, img_ref_feat):

        pre_offset = pre_offset_flow_sim[0]
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]
        
        pre_relu1_swapped_feat = self.flow_warp(img_ref_feat['relu1_1'], pre_flow['relu1_1'])
        pre_relu2_swapped_feat = self.flow_warp(img_ref_feat['relu2_1'], pre_flow['relu2_1'])
        pre_relu3_swapped_feat = self.flow_warp(img_ref_feat['relu3_1'], pre_flow['relu3_1'])

        # Unet
        x0 = self.unet_head(base)    # [B, 64, 160, 160]

        # -------------- Down ------------------
        # large scale
        down_relu1_offset = torch.cat([x0, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        down_relu1_offset = self.lrelu(self.down_large_offset_conv1(down_relu1_offset))
        down_relu1_offset = self.lrelu(self.down_large_offset_conv2(down_relu1_offset))
        down_relu1_swapped_feat = self.lrelu(
            self.down_large_dyn_agg([img_ref_feat['relu1_1'], down_relu1_offset],
                               pre_offset['relu1_1']))
        
        h = torch.cat([x0, down_relu1_swapped_feat], 1)
        h = self.down_head_large(h)
        h = self.ddf_block_1(h) + x0
        h = self.down_body_large(h) + x0
        x1 = self.down_tail_large(h)  # [B, 64, 80, 80]

        # medium scale
        down_relu2_offset = torch.cat([x1, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv1(down_relu2_offset))
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv2(down_relu2_offset))
        down_relu2_swapped_feat = self.lrelu(
            self.down_medium_dyn_agg([img_ref_feat['relu2_1'], down_relu2_offset],
                                pre_offset['relu2_1']))
        
        h = torch.cat([x1, down_relu2_swapped_feat], 1)
        h = self.down_head_medium(h)
        h = self.ddf_block_2(h) + x1
        h = self.down_body_medium(h) + x1
        x2 = self.down_tail_medium(h)    # [9, 64, 40, 40]

        # -------------- Up ------------------

        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([x2, pre_relu3_swapped_feat, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(
            self.up_small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset['relu3_1']))  
        
        x2_agg = torch.cat([sife, x2], dim=1)
        x2_agg = self.lrelu(self.high_q_conv1(x2_agg))

        # small scale
        h = torch.cat([x2_agg, relu3_swapped_feat], 1)
        h = self.up_head_small(h)
        h = self.ddf_block_3(h) + x2
        h = self.up_body_small(h) + x2
        x = self.up_tail_small(h)    # [9, 64, 80, 80]

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(
            self.up_medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                pre_offset['relu2_1']))
             
        # medium scale
        h = torch.cat([x+x1, relu2_swapped_feat], 1)
        h = self.up_head_medium(h)
        h = self.ddf_block_4(h) + x
        h = self.up_body_medium(h) + x
        x = self.up_tail_medium(h)   # [9, 64, 160, 160]

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(
            self.up_large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                               pre_offset['relu1_1']))
                      
        # large scale
        h = torch.cat([x+x0, relu1_swapped_feat], 1)
        h = self.up_head_large(h)
        h = self.ddf_block_5(h) + x
        h = self.up_body_large(h) + x
        x = self.up_tail_large(h)

        return x