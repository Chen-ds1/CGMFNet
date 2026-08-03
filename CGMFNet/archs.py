from torch import nn

from CMGE import CMGE
from MAAF import MAAF

__all__ = ['CGMFNetL4', 'CGMFNetL3', 'CGMFNetL2', 'CGMFNetL1']

class CGMFNetL4(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cmge0 = CMGE(input_channels, nb_filter[0], middle_channels=None, r=1)
        self.cmge1 = CMGE(nb_filter[0], nb_filter[1], middle_channels=None, r=1)
        self.cmge2 = CMGE(nb_filter[1], nb_filter[2], middle_channels=None, r=1)
        self.cmge3 = CMGE(nb_filter[2], nb_filter[3], middle_channels=None, r=1)
        self.cmge4 = CMGE(nb_filter[3], nb_filter[4], middle_channels=None, r=1)

        self.maaf0_1 = MAAF(nb_filter[0] + nb_filter[1], nb_filter[0], [nb_filter[0], nb_filter[1]])
        self.maaf1_1 = MAAF(nb_filter[1] + nb_filter[2], nb_filter[1], [nb_filter[1], nb_filter[2]])
        self.maaf2_1 = MAAF(nb_filter[2] + nb_filter[3], nb_filter[2], [nb_filter[2], nb_filter[3]])
        self.maaf3_1 = MAAF(nb_filter[3] + nb_filter[4], nb_filter[3], [nb_filter[3], nb_filter[4]])

        self.maaf0_2 = MAAF(nb_filter[0]*2 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[1]])
        self.maaf1_2 = MAAF(nb_filter[1]*2 + nb_filter[2], nb_filter[1],
                            [nb_filter[1], nb_filter[1], nb_filter[2]])
        self.maaf2_2 = MAAF(nb_filter[2]*2 + nb_filter[3], nb_filter[2],
                            [nb_filter[2], nb_filter[2], nb_filter[3]])

        self.maaf0_3 = MAAF(nb_filter[0]*3 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]])
        self.maaf1_3 = MAAF(nb_filter[1]*3 + nb_filter[2], nb_filter[1],
                            [nb_filter[1], nb_filter[1], nb_filter[1], nb_filter[2]])

        self.maaf0_4 = MAAF(nb_filter[0]*4 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, rgb_inputs, depth_inputs):

        [e_rgb0, e_depth0], fused0 = self.cmge0([rgb_inputs, depth_inputs])

        [e_rgb1, e_depth1], fused1 = self.cmge1([self.pool(e_rgb0), self.pool(e_depth0)])

        s0_1 = self.maaf0_1(fused0, self.up(fused1))

        [e_rgb2, e_depth2], fused2 = self.cmge2([self.pool(e_rgb1), self.pool(e_depth1)])

        s1_1 = self.maaf1_1(fused1, self.up(fused2))
        s0_2 = self.maaf0_2(fused0, s0_1, self.up(s1_1))

        [e_rgb3, e_depth3], fused3 = self.cmge3([self.pool(e_rgb2), self.pool(e_depth2)])

        s2_1 = self.maaf2_1(fused2, self.up(fused3))
        s1_2 = self.maaf1_2(fused1, s1_1, self.up(s2_1))
        s0_3 = self.maaf0_3(fused0, s0_1, s0_2, self.up(s1_2))

        [e_rgb4, e_depth4], fused4 = self.cmge4([self.pool(e_rgb3), self.pool(e_depth3)])

        s3_1 = self.maaf3_1(fused3, self.up(fused4))
        s2_2 = self.maaf2_2(fused2, s2_1, self.up(s3_1))
        s1_3 = self.maaf1_3(fused1, s1_1, s1_2, self.up(s2_2))
        s0_4 = self.maaf0_4(fused0, s0_1, s0_2, s0_3, self.up(s1_3))

        if self.deep_supervision:

            output1 = self.final1(s0_1)
            output2 = self.final2(s0_2)
            output3 = self.final3(s0_3)
            output4 = self.final4(s0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(s0_4)
            return output

class CGMFNetL3(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cmge0 = CMGE(input_channels, nb_filter[0], r=1)
        self.cmge1 = CMGE(nb_filter[0], nb_filter[1], r=1)
        self.cmge2 = CMGE(nb_filter[1], nb_filter[2], r=1)
        self.cmge3 = CMGE(nb_filter[2], nb_filter[3], r=1)

        self.maaf0_1 = MAAF(nb_filter[0] + nb_filter[1], nb_filter[0], [nb_filter[0], nb_filter[1]])
        self.maaf1_1 = MAAF(nb_filter[1] + nb_filter[2], nb_filter[1], [nb_filter[1], nb_filter[2]])
        self.maaf2_1 = MAAF(nb_filter[2] + nb_filter[3], nb_filter[2], [nb_filter[2], nb_filter[3]])

        self.maaf0_2 = MAAF(nb_filter[0]*2 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[1]])
        self.maaf1_2 = MAAF(nb_filter[1]*2 + nb_filter[2], nb_filter[1],
                            [nb_filter[1], nb_filter[1], nb_filter[2]])

        self.maaf0_3 = MAAF(nb_filter[0]*3 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[0], nb_filter[1]])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, rgb_inputs, depth_inputs):

        [e_rgb0, e_depth0], fused0 = self.cmge0([rgb_inputs, depth_inputs])
        [e_rgb1, e_depth1], fused1 = self.cmge1([self.pool(e_rgb0), self.pool(e_depth0)])
        [e_rgb2, e_depth2], fused2 = self.cmge2([self.pool(e_rgb1), self.pool(e_depth1)])
        [e_rgb3, e_depth3], fused3 = self.cmge3([self.pool(e_rgb2), self.pool(e_depth2)])

        s0_1 = self.maaf0_1(fused0, self.up(fused1))
        s1_1 = self.maaf1_1(fused1, self.up(fused2))
        s2_1 = self.maaf2_1(fused2, self.up(fused3))

        s0_2 = self.maaf0_2(fused0, s0_1, self.up(s1_1))
        s1_2 = self.maaf1_2(fused1, s1_1, self.up(s2_1))

        s0_3 = self.maaf0_3(fused0, s0_1, s0_2, self.up(s1_2))

        if self.deep_supervision:
            return [self.final1(s0_1), self.final2(s0_2), self.final3(s0_3)]
        else:
            return self.final(s0_3)

class CGMFNetL2(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cmge0 = CMGE(input_channels, nb_filter[0], r=1)
        self.cmge1 = CMGE(nb_filter[0], nb_filter[1], r=1)
        self.cmge2 = CMGE(nb_filter[1], nb_filter[2], r=1)

        self.maaf0_1 = MAAF(nb_filter[0] + nb_filter[1], nb_filter[0], [nb_filter[0], nb_filter[1]])
        self.maaf1_1 = MAAF(nb_filter[1] + nb_filter[2], nb_filter[1], [nb_filter[1], nb_filter[2]])

        self.maaf0_2 = MAAF(nb_filter[0]*2 + nb_filter[1], nb_filter[0],
                            [nb_filter[0], nb_filter[0], nb_filter[1]])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, rgb_inputs, depth_inputs):

        [e_rgb0, e_depth0], fused0 = self.cmge0([rgb_inputs, depth_inputs])
        [e_rgb1, e_depth1], fused1 = self.cmge1([self.pool(e_rgb0), self.pool(e_depth0)])
        [e_rgb2, e_depth2], fused2 = self.cmge2([self.pool(e_rgb1), self.pool(e_depth1)])

        s0_1 = self.maaf0_1(fused0, self.up(fused1))
        s1_1 = self.maaf1_1(fused1, self.up(fused2))

        s0_2 = self.maaf0_2(fused0, s0_1, self.up(s1_1))

        if self.deep_supervision:
            return [self.final1(s0_1), self.final2(s0_2)]
        else:
            return self.final(s0_2)

class CGMFNetL1(nn.Module):

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64]

        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.cmge0 = CMGE(input_channels, nb_filter[0], r=1)
        self.cmge1 = CMGE(nb_filter[0], nb_filter[1], r=1)

        self.maaf0_1 = MAAF(nb_filter[0] + nb_filter[1], nb_filter[0], [nb_filter[0], nb_filter[1]])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, rgb_inputs, depth_inputs):

        [e_rgb0, e_depth0], fused0 = self.cmge0([rgb_inputs, depth_inputs])
        [e_rgb1, e_depth1], fused1 = self.cmge1([self.pool(e_rgb0), self.pool(e_depth0)])

        s0_1 = self.maaf0_1(fused0, self.up(fused1))

        if self.deep_supervision:
            return [self.final1(s0_1)]
        else:
            return self.final(s0_1)
