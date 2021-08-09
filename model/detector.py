"""Defines the detector network structure."""
import torch
from torch import nn
from model.network import define_halve_unit, BottleneckBlock


class YetAnotherDarknet(nn.modules.Module):
    """Yet another darknet, imitating darknet-53 with depth of darknet-19."""

    def __init__(self, input_channel_size, depth_factor):
        super(YetAnotherDarknet, self).__init__()
        layers = []
        # 0
        layers += [nn.Conv2d(input_channel_size, depth_factor, kernel_size=3,
                             stride=1, padding=1, bias=False)]
        layers += [nn.BatchNorm2d(depth_factor)]
        layers += [nn.LeakyReLU(0.1)]
        layers += define_halve_unit(depth_factor)
        # 1,32*256*256
        depth_factor *= 2
        layers += [BottleneckBlock(depth_factor)]
        layers += define_halve_unit(depth_factor)
        layers += [nn.AvgPool2d(2,stride=2)]
        # 2,64*128*128
        depth_factor *= 2
        layers += [BottleneckBlock(depth_factor)]
        layers += [BottleneckBlock(depth_factor)]
        layers += define_halve_unit(depth_factor)
        layers += [nn.AvgPool2d(2, stride=4)]
        # 3,128*64*64
        depth_factor *= 2
        layers += [BottleneckBlock(depth_factor)]
        layers += [BottleneckBlock(depth_factor)]
        layers += define_halve_unit(depth_factor)
        layers += [nn.AvgPool2d(2, stride=4)]
        self.model = nn.Sequential(*layers)

    def forward(self, *x):
        return self.model(x[0])


class CameraPoseDetector(nn.modules.Module):
    """Detector for Camera Pose."""
    def __init__(self, input_channel_size, depth_factor, batch_size):
        super(CameraPoseDetector, self).__init__()
        self.extract_feature = YetAnotherDarknet(input_channel_size,
                                                 depth_factor)
        self.relu = nn.LeakyReLU(0.1, inplace=False)
        self.fc1 = nn.Linear(in_features=4*256*1*1,out_features=1000)
        self.fc2 = nn.Linear(in_features=1 * 1000, out_features=48)
        self.dropout = nn.Dropout2d(0.1)
        self.batch_size = batch_size

    def forward(self, x):
        prediction = []
        features_all = self.relu(self.dropout(self.extract_feature(x)))
        features_reshape = features_all.reshape(round(len(x)/4),4,256,1,1)
        features_concat = torch.cat([features_reshape[:,0],features_reshape[:,1],features_reshape[:,2],features_reshape[:,3]],dim=1)#3*1024*1?
        for index in range(len(features_concat)):
            fc1 = self.fc1(self.relu(features_concat[index].view(1,4*256)))
            fc2 = self.fc2(self.relu(fc1))
            part_r, part_t = torch.split(fc2, [36, 12], dim=1)#dim?
            part_r = torch.tanh(part_r)
            batch_prediction = torch.cat([part_r, part_t], dim=1)
            prediction.append(batch_prediction)
        return torch.cat(prediction)
