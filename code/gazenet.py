import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import resnet as M
import resnet_fpn as resnet_fpn


# Feature Pyramid Network
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # bottom up
        self.resnet = resnet_fpn.resnet50(pretrained=True)

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c5_conv = nn.Conv2d(2048, 256, (1, 1))
        self.c4_conv = nn.Conv2d(1024, 256, (1, 1))
        self.c3_conv = nn.Conv2d(512, 256, (1, 1))
        self.c2_conv = nn.Conv2d(256, 256, (1, 1))
        #self.max_pool = nn.MaxPool2d((1, 1), stride=2)

        self.p5_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p4_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p3_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)

        # predict heatmap
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=1)
 
    def top_down(self, x):
        c2, c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p4 = self.upsample(p5) + self.c4_conv(c4)
        p3 = self.upsample(p4) + self.c3_conv(c3)
        p2 = self.upsample(p3) + self.c2_conv(c2)

        p5 = self.relu(self.p5_conv(p5))
        p4 = self.relu(self.p4_conv(p4))
        p3 = self.relu(self.p3_conv(p3))
        p2 = self.relu(self.p2_conv(p2))

        return p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # top down
        p2, p3, p4, p5 = self.top_down((c2, c3, c4, c5))

        heatmap = self.sigmoid(self.predict(p2))
        return heatmap


class GazeNet(nn.Module):
    def __init__(self):
        super(GazeNet, self).__init__()
        self.face_net = M.resnet50(pretrained=True)
        self.face_process = nn.Sequential(nn.Linear(2048, 512),
                                          nn.ReLU(inplace=True))

        self.fpn_net = FPN()

        self.eye_position_transform = nn.Sequential(nn.Linear(2, 256),
                                                    nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(nn.Linear(512 + 256, 256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, 2))

        self.relu = nn.ReLU(inplace=False)
       
        # change first conv layer for fpn_net because we concatenate 
        # multi-scale gaze field with image image 
        conv = [x.clone() for x in self.fpn_net.resnet.conv1.parameters()][0]
        new_kernel_channel = conv.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        new_kernel = torch.cat((conv.data, new_kernel_channel), 1)
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        new_conv.weight.data = new_kernel
        self.fpn_net.resnet.conv1 = new_conv

    def forward(self, x):
        image, face_image, gaze_field, eye_position = x
        # face part forward
        face_feature = self.face_net(face_image)
        face_feature = self.face_process(face_feature)

        # eye position transform
        eye_feature = self.eye_position_transform(eye_position)

        # fusion
        feature = torch.cat((face_feature, eye_feature), 1)
        direction = self.fusion(feature)

        # infer gaze direction and normalized
        norm = torch.norm(direction, 2, dim=1)
        normalized_direction = direction / norm.view([-1, 1])

        # generate gaze field map
        batch_size, channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        gaze_field = gaze_field.view([batch_size, -1, 2])
        gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, 2, 1]))
        gaze_field_map = gaze_field.view([batch_size, height, width, 1])
        gaze_field_map = gaze_field_map.permute([0, 3, 1, 2]).contiguous()

        gaze_field_map = self.relu(gaze_field_map)
        #print gaze_field_map.size()

        # mask with gaze_field
        gaze_field_map_2 = torch.pow(gaze_field_map, 2)
        gaze_field_map_3 = torch.pow(gaze_field_map, 5)
        image = torch.cat([image, gaze_field_map, gaze_field_map_2, gaze_field_map_3], dim=1)
        heatmap = self.fpn_net(image)

        return direction, heatmap

