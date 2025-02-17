import torch
import torch.nn as nn
from .basic_blocks import *
from .resnet import ResNet34
from .inception_transformer import iformer_small_384

class DRENet(nn.Module):
    def __init__(self,type):
        super(DRENet, self).__init__()
        self.loss = nn.BCELoss()
        self.sat = iformer_small_384(True)
        self.add = iformer_small_384(True)
        self.ske_backbone = ResNet34()
        self.patch1 = self.sat.patch_embed
        self.encoder1 = self.sat.blocks1
        self.patch2 = self.sat.patch_embed2
        self.encoder2 = self.sat.blocks2
        self.patch3 = self.sat.patch_embed3
        self.encoder3 = self.sat.blocks3
        self.patch4 = self.sat.patch_embed4
        self.encoder4 = self.sat.blocks4
        #
        self.patch1_add = self.add.patch_embed
        self.encoder1_add = self.add.blocks1
        self.patch2_add = self.add.patch_embed2
        self.encoder2_add = self.add.blocks2
        self.patch3_add = self.add.patch_embed3
        self.encoder3_add = self.add.blocks3
        self.patch4_add = self.add.patch_embed4
        self.encoder4_add = self.add.blocks4
        channels = [96, 192, 320, 384]
        resnet_channels = [64, 128, 256, 512]
        self.fpn = FeaturePyramidNetwork(channels)
        self.lsk = LSKblock(channels[0])
        self.fine = SkeletonNetwork(channels)
        self.ske_net = SkeletonNetwork(channels)
        
        self.fuse4 = FusionModule(channels[3],channels[3])
        self.fuse3 = FusionModule(channels[2],channels[2])
        self.fuse2 = FusionModule(channels[1],channels[1])
        self.fuse1 = FusionModule(channels[0],channels[0])

        self.enc_fuse = EncoderFusion(channels, channels[0], 128, 128)
        self.enc_fuse_ske = EncoderFusion(channels, resnet_channels[3], 16, 16)
        self.trans = nn.Sequential(
             nn.Conv2d(resnet_channels[0], channels[0], 1, bias=False),
             nn.BatchNorm2d(channels[0]),
             nn.ReLU(inplace=True)
         )
        self.de4 = DecoderBlock(channels[3], channels[2])
        self.de3 = DecoderBlock(channels[2], channels[1])
        self.de2 = DecoderBlock(channels[1], channels[0])
        self.de1 = DecoderBlock(channels[0], channels[0])
        #
        self.ske_de4 = DecoderBlock(resnet_channels[3], resnet_channels[2])
        self.ske_de3 = DecoderBlock(resnet_channels[2], resnet_channels[1])
        self.ske_de2 = DecoderBlock(resnet_channels[1], resnet_channels[0])
        self.ske_de1 = DecoderBlock(resnet_channels[0], resnet_channels[0])

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0] // 2, channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0] // 2, 1, 1)
        )

        self.ske_upsample = nn.Sequential(
            nn.Conv2d(resnet_channels[0], resnet_channels[0] // 2, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(resnet_channels[0] // 2, resnet_channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(resnet_channels[0] // 2, 1, 1)
        )
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(channels[0], channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0] // 2, channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0] // 2, 1, 1)
        )

        self.satellite_upsample = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[0] // 2, channels[0] // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0] // 2, 1, 1)
        )


    def forward(self,inputs):
        x = inputs[:, :3, :, :]
        add = inputs[:, 3:, :, :]
        x = self.patch1(x)
        add = self.patch1_add(add)

        x_e1 = self.encoder1(x)
        add_e1 = self.encoder1_add(add)
        x_e1, add_e1 = x_e1.permute(0, 3, 1, 2),add_e1.permute(0, 3, 1, 2)

        x_e2 = self.encoder2(self.patch2(x_e1))
        add_e2 = self.encoder2_add(self.patch2_add(add_e1))
        x_e2, add_e2 = x_e2.permute(0, 3, 1, 2),add_e2.permute(0, 3, 1, 2)

        x_e3 = self.encoder3(self.patch3(x_e2))
        add_e3 = self.encoder3_add(self.patch3_add(add_e2))
        x_e3, add_e3 = x_e3.permute(0, 3, 1, 2),add_e3.permute(0, 3, 1, 2)

        x_e4 = self.encoder4(self.patch4(x_e3))
        add_e4 = self.encoder4_add(self.patch4_add(add_e3))
        x_e4, add_e4 = x_e4.permute(0, 3, 1, 2),add_e4.permute(0, 3, 1, 2)
        
        fuse1 = self.fuse1(x_e1, add_e1)
        fuse2 = self.fuse2(x_e2, add_e2)
        fuse3 = self.fuse3(x_e3, add_e3)
        fuse4 = self.fuse4(x_e4, add_e4)

        road = self.enc_fuse(fuse1, fuse2, fuse3, fuse4)
        skeleton = self.enc_fuse_ske(fuse1, fuse2, fuse3, fuse4)

        road = self.fine(road)
        
        road_1 = road
        
        road_1 = self.upsample_1(road_1)


        skeleton = self.ske_backbone.layer4(skeleton)
        de_ske4 = self.ske_de4(skeleton)
        de_ske4 = self.ske_backbone.layer3(de_ske4)
        de_ske3 = self.ske_de3(de_ske4)
        de_ske3 = self.ske_backbone.layer2(de_ske3)
        de_ske2 = self.ske_de2(de_ske3)
        de_ske2 = self.ske_backbone.layer1(de_ske2)
        de_ske1 = self.ske_de1(de_ske2)

        road = self.lsk(road, self.trans(de_ske2))
        
        #road = road + self.trans(de_ske2)
        road = self.ske_net(road)
        road = self.upsample(road)

        skeleton_output = self.ske_upsample(de_ske1)

        # fuse_skeleton = self.lsk(skeleton2)



        x = self.de4(x_e4) + x_e3
        x = self.de3(x) + x_e2
        x = self.de2(x) + x_e1
        x = self.de1(x)
        sat_out = self.satellite_upsample(x)

        return torch.sigmoid(road), torch.sigmoid(skeleton_output), torch.sigmoid(sat_out), torch.sigmoid(road_1)








