import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu, inplace=True)
from .CBAM import ChannelAttention

class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, padding=2)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(4, 1, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.fuse = ConvBNReLu(dim*2, dim, 3, stride=1, padding=1)

    def forward(self, x, skeleton):
        attn1 = self.conv0(skeleton)
        avg_attn1 = torch.mean(attn1, dim=1, keepdim=True)
        max_attn1, _ = torch.max(attn1, dim=1, keepdim=True)
        attn2 = self.conv_spatial(attn1)
        # attn2 = attn1

        attn1 = self.conv1(attn1)
        x_attn2 = self.conv2(x)

        #attn = torch.cat([attn1, x_attn2], dim=1)
        avg_attn = torch.mean(x_attn2, dim=1, keepdim=True)
        max_attn, _ = torch.max(x_attn2, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn, max_attn1, avg_attn1], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        feature = sig * attn2
        out = self.fuse(torch.cat([feature, x],dim=1))
        return out

class EncoderFusion(nn.Module):
    def __init__(self, channels, outchannel, size_h, size_w):
        super(EncoderFusion, self).__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.conv1 = ConvBNReLu(channels[0], channels[0], 1, bias=False)
        self.conv2 = ConvBNReLu(channels[1], channels[0], 1, bias=False)
        self.conv3 = ConvBNReLu(channels[2], channels[0], 1, bias=False)
        self.conv4 = ConvBNReLu(channels[3], channels[0], 1, bias=False)
        self.fuse = ConvBNReLu(channels[0]*4, outchannel, kernel_size=3, padding=1)

    def forward(self, x1,x2,x3,x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x1 = F.interpolate(x1, size=(self.size_h, self.size_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(self.size_h, self.size_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(self.size_h, self.size_w), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(self.size_h, self.size_w), mode='bilinear', align_corners=True)
        out = torch.cat([x1,x2,x3,x4], dim=1)
        out = self.fuse(out)
        return out


class SkeletonNetwork(nn.Module):
    def __init__(self, channels):
        super(SkeletonNetwork, self).__init__()
        self.down1 = ConvBNReLu(channels[0], channels[1], 3, stride=1, padding=1)
        self.down2 = ConvBNReLu(channels[1], channels[0], 3, stride=1, padding=1)
        self.down3 = ConvBNReLu(channels[0], channels[1], 3, stride=1, padding=1)
        self.down4 = ConvBNReLu(channels[1], channels[0], 3, stride=1, padding=1)
        # self.up1 = TransConvBNReLu(channels[2], channels[1], kernel_size=4, stride=4)
        # self.up2 = TransConvBNReLu(channels[1], channels[0], kernel_size=4, stride=4)

    def forward(self, x):
        res = x
        x = self.down1(x)
        x = self.down2(x) + res
        res = x
        x = self.down3(x)
        x = self.down4(x) + res
        # x = self.up1(x)
        # x = self.up2(x)
        return x

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.indice_channel = channel // 3
        last_channel = channel - self.indice_channel * 2
        self.conv1 = nn.Conv2d(channel, self.indice_channel, 1)
        self.conv_shape2 = nn.Conv2d(channel, self.indice_channel, kernel_size=3, dilation=2, padding=2)
        self.conv_shape3 = nn.Conv2d(channel, last_channel, kernel_size=3, dilation=4, padding=4)

    def forward(self, x):
        x = self.bn(x)
        x1 = self.conv1(x)
        x2 = self.conv_shape2(x)
        x3 = self.conv_shape3(x)
        return torch.cat([x1, x2, x3], dim=1)

class QKV(nn.Module):
    def __init__(self, channel):
        super(QKV, self).__init__()
        self.q = Attention(channel)
        self.k = Attention(channel)
        self.v = Attention(channel)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q,k,v

class FusionModule(nn.Module):
    def __init__(self, sat_channel, traj_channel):
        super(FusionModule, self).__init__()
        self.half = sat_channel
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(sat_channel+traj_channel),
            nn.Conv2d(sat_channel+traj_channel, sat_channel, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fuseq = nn.Conv2d(sat_channel*2, sat_channel*2, 1, bias=False)
        self.qkv_sat = QKV(sat_channel)
        self.qkv_traj = QKV(traj_channel)

    def forward(self, sat, traj):
        q_sat, k_sat, v_sat = self.qkv_sat(sat)
        q_traj, k_traj, v_traj = self.qkv_traj(traj)
        fuseq = self.fuseq(torch.cat([q_sat, q_traj], dim=1))
        q_s, q_t = fuseq[:, :self.half, :, :], fuseq[:, self.half:, :, :]
        attn_sat = torch.sigmoid(k_sat * q_s) * v_sat
        attn_traj = torch.sigmoid(k_traj * q_t) * v_traj
        fuse = self.fuse(torch.cat([attn_sat, attn_traj], dim=1))
        return fuse


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=True):
        super(ConvBNReLu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

class TransConvBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=0, norm_layer=nn.BatchNorm2d, bias=True):
        super(TransConvBNReLu, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )


class SkipConnect(nn.Module):
    def __init__(self,catchannels,channels):
        super(SkipConnect, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels + catchannels, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(4)

    def forward(self,x,prex):
        _,_,h,w = x.shape
        _,_,ph,pw = prex.shape
        if h < ph:
            prex = self.pool(prex)
        elif h > ph:
            prex = F.interpolate(prex, size=(h,w),mode='bilinear')
        x = torch.cat([x,prex],dim=1)
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_filters, 1)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nonlinearity
        self.res = nn.Sequential(
             nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
             nn.BatchNorm2d(n_filters),
             nn.ReLU(inplace=True),
             nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
             nn.BatchNorm2d(n_filters)
        )
        self.deconv2 = nn.ConvTranspose2d(
            n_filters, n_filters, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(n_filters, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity
        # self.dilate1 = nn.Sequential(
        #   nn.Conv2d(n_filters, n_filters, 3, dilation=2, padding=2),
        #   nn.BatchNorm2d(n_filters),
        #   nn.ReLU(inplace=True)
        # )
        # self.dilate2 = nn.Sequential(
        #   nn.Conv2d(n_filters, n_filters, 3, dilation=2, padding=2),
        #   nn.BatchNorm2d(n_filters),
        #   nn.ReLU(inplace=True)
        # )
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        #x = F.relu(self.res(x) + x)
        #d = self.dilate1(x)
        #x = self.dilate2(d) + x
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class FDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(FDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, 1)
        self.norm1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nonlinearity
        self.res = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters)
        )
        self.deconv2 = nn.ConvTranspose2d(
            n_filters, n_filters, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(n_filters, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity


    def forward(self, x):
        #x = self.pre_norm(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = F.relu(self.res(x) + x)
        # d = self.dilate1(x)
        # x = self.dilate2(d) + x
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        # x = self.conv3(x)
        # x = self.norm3(x)
        # x = self.relu3(x)
        return x


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.dilate1 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(
            channel, channel, kernel_size=3, dilation=8, padding=8)

        # self.spm = SPBlock(channel, channel, norm_layer=nn.BatchNorm2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        dilate1_out = nonlinearity(self.dilate1(x))

        dilate2_out = nonlinearity(self.dilate2(dilate1_out))

        dilate3_out = nonlinearity(self.dilate3(dilate2_out))

        dilate4_out = nonlinearity(self.dilate4(dilate3_out))

        # spm_out = self.spm(x)

        # out = (x + dilate1_out + dilate2_out + dilate3_out + dilate4_out)*spm_out
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock1DConv2(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock1DConv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv1 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x = torch.cat((x1, x2), 1)

        x = F.interpolate(x, scale_factor=2)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DecoderBlock1DConv4(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock1DConv4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv1 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 16, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 16, (1, 9), padding=(0, 4)
        )

        self.norm2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 8)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 8, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = F.interpolate(x, scale_factor=2)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# class FeaturePyramidNetwork(nn.Module):
#     def __init__(self, channels):
#         super(FeaturePyramidNetwork, self).__init__()
#         self.traj_conv1 = ConvBNReLu(3, channels[0], kernel_size=4, stride=4)
#         self.traj_conv2 = ConvBNReLu(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
#         self.traj_conv3 = ConvBNReLu(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
#         self.traj_conv4 = ConvBNReLu(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
#         self.fuse1 = ConvBNReLu(channels[3], channels[2], kernel_size=3, stride=1, padding=1)
#         self.fuse2 = ConvBNReLu(channels[2], channels[1], kernel_size=3, stride=1, padding=1)
#         self.fuse3 = ConvBNReLu(channels[1], channels[0], kernel_size=3, stride=1, padding=1)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#
#     def forward(self, inputs, traj):
#         traj1 = self.traj_conv1(traj)
#         traj2 = self.traj_conv2(traj1)
#         traj3 = self.traj_conv3(traj2)
#         traj4 = self.traj_conv4(traj3)
#
#         fuse = traj4 + inputs[3]
#         fuse = self.fuse1(fuse)
#         fuse1 = self.upsample(fuse)
#         fuse2 = traj3 + inputs[2] + fuse1
#         fuse2 = self.fuse2(fuse2)
#         fuse2 = self.upsample(fuse2)
#         fuse3 = traj2 + inputs[1] + fuse2
#         fuse3 = self.fuse3(fuse3)
#         fuse3 = self.upsample(fuse3)
#         fuse4 = traj1 + inputs[0] + fuse3
#         # fuse4 = self.upsample(fuse4)
#
#         return fuse4

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, channels):
        super(FeaturePyramidNetwork, self).__init__()
        # self.traj_conv1 = ConvBNReLu(3, channels[0], kernel_size=4, stride=4)
        # self.traj_conv2 = ConvBNReLu(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        # self.traj_conv3 = ConvBNReLu(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        # self.traj_conv4 = ConvBNReLu(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        self.fuse1 = ConvBNReLu(channels[3], channels[2], kernel_size=3, stride=1, padding=1)
        self.fuse2 = ConvBNReLu(channels[2], channels[1], kernel_size=3, stride=1, padding=1)
        self.fuse3 = ConvBNReLu(channels[1], channels[0], kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, traj):
        # traj1 = self.traj_conv1(traj)
        # traj2 = self.traj_conv2(traj1)
        # traj3 = self.traj_conv3(traj2)
        # traj4 = self.traj_conv4(traj3)

        fuse = traj[3] + inputs[3]
        fuse = self.fuse1(fuse)
        fuse1 = self.upsample(fuse)
        fuse2 = traj[2] + inputs[2] + fuse1
        fuse2 = self.fuse2(fuse2)
        fuse2 = self.upsample(fuse2)
        fuse3 = traj[1] + inputs[1] + fuse2
        fuse3 = self.fuse3(fuse3)
        fuse3 = self.upsample(fuse3)
        fuse4 = traj[0] + inputs[0] + fuse3
        # fuse = self.upsample(fuse3) + traj[0] + inputs[0]

        return fuse4