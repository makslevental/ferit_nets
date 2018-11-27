import pdb  # pdb.set_trace()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.nn import SELU
from torch.autograd import Variable
# from MNN_New2D import MNN
import gzip

#
# class init_weights():
#     def __init__(self, init_function):
#         self.init_func = getattr(torch.nn.init, init_function)
#
#     def weight_init(self, m):
#         if (isinstance(m, torch.nn.Linear) | isinstance(m, torch.nn.Conv2d) | isinstance(m, torch.nn.Conv3d)):
#             self.init_func(m.weight.data)


def compress_layers(net):
    items = list(net.items())
    compression_sizes = np.zeros([len(items)])
    decompression_sizes = np.zeros([len(items)])
    for layer in range(len(items)):
        key = items[layer][0]
        print(key, end='\t')
        values = (items[layer][1].numpy() * 100).round()
        values = gzip.compress(values)
        compression_sizes[layer] = len(values)
        decompression_sizes[layer] = len(gzip.decompress(values))
        print('{}\t{}'.format(int(decompression_sizes[layer]), int(compression_sizes[layer])))
    print('total \t\t{}\t{}'.format(int(sum(decompression_sizes)), int(sum(compression_sizes))))


def parameter_counter(net):
    # net = net()
    par = list(net.parameters())
    total = 0
    for i in range(len(par)):
        x = par[i]
        s = x.size()
        total += np.prod(s)
        print(np.prod(s))
    print('The model has ' + str(total) + ' parameters')
    return


class Auto_GPR_15_300(nn.Module):
    def __init__(self):
        super(Auto_GPR_15_300, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=10, stride=5)
        self.conv2 = nn.Conv1d(4, 8, 6, 5)
        self.enc1 = nn.Linear(88, 10)
        self.dec1 = nn.Linear(10, 88)
        self.dec2 = nn.Linear(88, 300)

    def forward(self, y):
        # pdb.set_trace()
        x = y.mean(dim=1).mean(dim=2).squeeze(1)  # mean Ascan
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * 11)
        z = self.enc1(x)
        x = F.relu(z)
        x = F.relu(self.dec1(x))
        x = self.dec2(x)
        return x


class GPR_15_300(nn.Module):
    def __init__(self):
        super(GPR_15_300, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 5))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 5))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()                          #x: 1x1x15x15x460
        x = x.unsqueeze(1)  # new gen
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = F.relu(self.conv4(x))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Auto_Ascan(nn.Module):
    def __init__(self):
        super(Auto_Ascan, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=10, stride=5)
        self.conv2 = nn.Conv1d(4, 8, 6, 5)
        self.enc1 = nn.Linear(88, 10)
        self.dec1 = nn.Linear(10, 88)
        self.dec2 = nn.Linear(88, 300)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.squeeze(1)
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = x.view(-1, 8 * 11)
        z = self.enc1(x)
        x = F.selu(z)
        z = self.smax(z)
        x = F.selu(self.dec1(x))
        x = self.dec2(x)
        return x, z


class zip300(nn.Module):
    def __init__(self):
        super(zip300, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (6, 6, 8), (1, 1, 2))
        self.conv2 = nn.Conv3d(6, 12, (5, 5, 6))
        self.conv3 = nn.Conv3d(12, 24, (4, 4, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 5))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()                          #x: 1x1x15x15x460
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, 0


class GPR300(nn.Module):
    def __init__(self):
        super(GPR300, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (6, 6, 8), (1, 1, 2))
        self.conv2 = nn.Conv3d(6, 12, (5, 5, 6))
        self.conv3 = nn.Conv3d(12, 24, (4, 4, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 5))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()                          #x: 1x1x15x15x460
        x = x.unsqueeze(1)  # new gen
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Feat128(nn.Module):
    def __init__(self):
        super(Feat128, self).__init__()
        self.fc1 = nn.Linear(128, 32)
        self.fc0 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc0(x)
        return x, 0


class MNN3(nn.Module):
    def __init__(self):
        super(MNN3, self).__init__()
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.MNN1 = MNN(1, 5, 5)
        self.MNN2 = MNN(5, 2, 5)
        self.fc1 = nn.Linear(5 * 2 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.squeeze(2)
        x = self.pool(x)
        x = self.MNN1(x)
        x = self.MNN2(x)
        x = x.view(-1, 5 * 2 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, 0


class B90F(nn.Module):
    def __init__(self):
        super(B90F, self).__init__()
        self.conv1c = nn.Conv2d(1, 4, (6, 8), (1, 3))
        self.conv2c = nn.Conv2d(4, 8, (5, 8))
        self.conv3c = nn.Conv2d(8, 12, (4, 8))
        self.conv4c = nn.Conv2d(12, 24, (4, 8))
        self.conv1d = nn.Conv2d(1, 4, (6, 8), (1, 3))
        self.conv2d = nn.Conv2d(4, 8, (5, 8))
        self.conv3d = nn.Conv2d(8, 12, (4, 8))
        self.conv4d = nn.Conv2d(12, 24, (4, 8))
        self.fcc0 = nn.Linear(24 * 4 * 7, 256)
        self.fcd0 = nn.Linear(24 * 4 * 7, 256)
        self.fcc1 = nn.Linear(256, 64)
        self.fcd1 = nn.Linear(256, 64)
        self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        ct = x[:, 0, :, :, :]
        ct = ct.squeeze(2)
        dt = x[:, 1, :, :, :]
        dt = dt.squeeze(2)
        del x
        ct = F.relu(self.conv1c(ct))
        ct = F.relu(self.conv2c(ct))
        ct = F.relu(self.conv3c(ct))
        ct = F.relu(self.conv4c(ct))
        ct = ct.view(-1, 24 * 4 * 7)
        dt = F.relu(self.conv1d(dt))
        dt = F.relu(self.conv2d(dt))
        dt = F.relu(self.conv3d(dt))
        dt = F.relu(self.conv4d(dt))
        dt = dt.view(-1, 24 * 4 * 7)
        ct = F.relu(self.fcc0(ct))
        dt = F.relu(self.fcd0(dt))
        ct = F.relu(self.fcc1(ct))
        dt = F.relu(self.fcd1(dt))
        x = torch.cat([ct, dt], dim=1)
        del ct, dt
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class B90Fm(nn.Module):
    def __init__(self):
        super(B90Fm, self).__init__()
        self.conv1c = nn.Conv2d(1, 4, (6, 8), (1, 3))
        self.conv2c = nn.Conv2d(4, 8, (5, 8))
        self.conv3c = nn.Conv2d(8, 12, (4, 8))
        self.conv4c = nn.Conv2d(12, 24, (4, 8))
        self.conv1d = nn.Conv2d(1, 4, (6, 8), (1, 3))
        self.conv2d = nn.Conv2d(4, 8, (5, 8))
        self.conv3d = nn.Conv2d(8, 12, (4, 8))
        self.conv4d = nn.Conv2d(12, 24, (4, 8))
        self.fcc0 = nn.Linear(24 * 4 * 7, 256)
        self.fcd0 = nn.Linear(24 * 4 * 7, 256)
        self.fcc1 = nn.Linear(256, 64)
        self.fcd1 = nn.Linear(256, 64)
        self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        out = Variable(torch.zeros(x.size(0), 50)).cuda()
        numero = 0
        for i in range(5):
            for j in range(5):
                ct = x[:, 0, i, :, :]
                dt = x[:, 1, i, :, :]
                dt = dt.unsqueeze(1)
                ct = ct.unsqueeze(1)
                ct = F.relu(self.conv1c(ct))
                ct = F.relu(self.conv2c(ct))
                ct = F.relu(self.conv3c(ct))
                ct = F.relu(self.conv4c(ct))
                ct = ct.view(-1, 24 * 4 * 7)
                dt = F.relu(self.conv1d(dt))
                dt = F.relu(self.conv2d(dt))
                dt = F.relu(self.conv3d(dt))
                dt = F.relu(self.conv4d(dt))
                dt = dt.view(-1, 24 * 4 * 7)
                ct = F.relu(self.fcc0(ct))
                dt = F.relu(self.fcd0(dt))
                ct = F.relu(self.fcc1(ct))
                dt = F.relu(self.fcd1(dt))
                y = torch.cat([ct, dt], dim=1)
                del ct, dt
                y = F.relu(self.fc0(y))
                y = F.relu(self.fc1(y))
                y = self.fc2(y)
                out[:, numero:numero + 2] = y
                numero += 2
        return out, 0


class B90F_MNN(nn.Module):
    def __init__(self):
        super(B90F, self).__init__()
        self.MNN1c = MNN(1, 5, 5)
        self.MNN2c = MNN(5, 2, 5)
        self.MNN1d = MNN(1, 5, 5)
        self.MNN2d = MNN(5, 2, 5)
        self.fc0 = nn.Linear(1344 * 6, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        ct0 = x[:, 0, :, :, :]
        ct0 = ct0.squeeze(2)
        dt0 = x[:, 1, :, :, :]
        dt0 = dt0.squeeze(2)
        x = Variable(torch.zeros(x.size(0), 1344 * 6)).cuda()
        # slide 12, cut 19x19, total 6 frames
        for frame_no in range(6):
            ct = ct0[:, :, :, 19 * frame_no:19 * frame_no + 12]
            dt = dt0[:, :, :, 19 * frame_no:19 * frame_no + 12]
            ct = self.MNN1c(ct)
            ct = self.MNN2c(ct)
            dt = self.MNN1d(dt)
            dt = self.MNN2d(dt)
            ct = ct.view(-1, 24 * 4 * 7)
            dt = dt.view(-1, 24 * 4 * 7)
            x[:, 1344 * frame_no:1344 * (frame_no + 1)] = torch.cat([ct, dt], dim=1)

        del ct0, dt0
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, 0


class MNN0(nn.Module):
    def __init__(self):
        super(MNN0, self).__init__()
        self.pool = nn.MaxPool2d((1, 4), (1, 4))
        self.MNN1 = MNN(1, 5, 7)
        # self.conv1 = nn.Conv2d(2, 4, kernel_size=5)
        self.MNN2 = MNN(5, 2, 5)
        # self.MNN3 = MNN(5,1,5)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(5 * 2 * 9 * 9, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.squeeze(2)
        x = self.pool(x)
        x = x[:, :, :, 1:20]
        output = self.MNN1(x)
        output = self.MNN2(output)
        # output = self.MNN2(output)
        # output = self.MNN3(output)
        output = output.view(-1, 5 * 2 * 9 * 9)
        output = F.relu(self.fc1(output))
        # output = F.dropout(output, training=self.training)
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output, 0


class MNN1(nn.Module):
    def __init__(self):
        super(MNN1, self).__init__()
        self.pool = nn.MaxPool2d((1, 4), (1, 4))
        self.MNN1 = MNN(1, 6, 7)
        # self.conv1 = nn.Conv2d(2, 4, kernel_size=5)
        self.MNN2 = MNN(6, 2, 5)
        self.MNN3 = MNN(12, 2, 5)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(12 * 2 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = x.squeeze(2)
        x = self.pool(x)
        x = x[:, :, :, 1:20]
        output = self.MNN1(x)
        output = self.MNN2(output)
        output = self.MNN3(output)
        # output = self.MNN2(output)
        # output = self.MNN3(output)
        output = output.view(-1, 12 * 2 * 5 * 5)
        output = F.relu(self.fc1(output))
        # output = F.dropout(output, training=self.training)
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output, 0


class Feat128_4(nn.Module):
    def __init__(self):
        super(Feat128_4, self).__init__()
        self.fc4 = nn.Linear(128 * 4, 128 * 3)
        self.fc3 = nn.Linear(128 * 3, 128 * 2)
        self.fc2 = nn.Linear(128 * 2, 128 * 1)
        self.fc1 = nn.Linear(128, 32)
        self.fc0 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = self.fc0(x)
        return x, 0


class Feat128_3(nn.Module):
    def __init__(self):
        super(Feat128_3, self).__init__()
        self.fc3 = nn.Linear(128 * 3, 128 * 2)
        self.fc2 = nn.Linear(128 * 2, 128 * 1)
        self.fc1 = nn.Linear(128, 32)
        self.fc0 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = self.fc0(x)
        return x, 0


class Feat300_128_2(nn.Module):
    def __init__(self):
        super(Feat300_128_2, self).__init__()
        self.fc4 = nn.Linear(300 + 128 * 2, 128 * 3)
        self.fc3 = nn.Linear(128 * 3, 128 * 2)
        self.fc2 = nn.Linear(128 * 2, 128 * 1)
        self.fc1 = nn.Linear(128, 32)
        self.fc0 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = self.fc0(x)
        return x, 0


class Feat128_2(nn.Module):
    def __init__(self):
        super(Feat128_2, self).__init__()
        self.fc2 = nn.Linear(128 * 2, 128 * 1)
        self.fc1 = nn.Linear(128, 32)
        self.fc0 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = self.fc0(x)
        return x, 0


class GPR4(nn.Module):
    def __init__(self):
        super(GPR4, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


class B90(nn.Module):
    def __init__(self):
        super(B90, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (6, 8), (1, 3))
        self.conv2 = nn.Conv2d(4, 8, (5, 8))
        self.conv3 = nn.Conv2d(8, 12, (4, 8))
        self.conv4 = nn.Conv2d(12, 24, (4, 8))
        self.fc1 = nn.Linear(24 * 4 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = F.relu(self.conv2(x))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = F.relu(self.conv3(x))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = F.relu(self.conv4(x))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 24 * 4 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


class B90_MNN(nn.Module):
    def __init__(self):
        super(B90_MNN, self).__init__()
        self.MNN1 = MNN(1, 4, 7)
        self.MNN2 = MNN(4, 8, (5, 8))
        self.MNN3 = MNN(8, 12, (4, 8))
        self.MNN4 = MNN(12, 24, (4, 8))
        self.fc1 = nn.Linear(24 * 4 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = F.relu(self.conv1(x))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = F.relu(self.conv2(x))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = F.relu(self.conv3(x))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = F.relu(self.conv4(x))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 24 * 4 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


class GPR90(nn.Module):
    def __init__(self):
        super(GPR90, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, (6, 6, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(4, 8, (5, 5, 8))
        self.conv3 = nn.Conv3d(8, 12, (4, 4, 8))
        self.conv4 = nn.Conv3d(12, 24, (4, 4, 8))
        self.fc1 = nn.Linear(24 * 4 * 4 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = F.relu(self.conv1(x))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = F.relu(self.conv2(x))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = F.relu(self.conv3(x))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = F.relu(self.conv4(x))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 24 * 4 * 4 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


class GPR90v2(nn.Module):
    def __init__(self):
        super(GPR90v2, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (6, 6, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (5, 5, 8))
        self.conv3 = nn.Conv3d(12, 24, (4, 4, 8))
        self.conv4 = nn.Conv3d(24, 48, (4, 4, 8))
        self.fc0 = nn.Linear(48 * 4 * 4 * 7, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = F.relu(self.conv1(x))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = F.relu(self.conv2(x))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = F.relu(self.conv3(x))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = F.relu(self.conv4(x))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 4 * 4 * 7)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, y


class Slice4(nn.Module):
    def __init__(self):
        super(Slice4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (8, 16))
        self.conv2 = nn.Conv2d(4, 8, (4, 6))
        self.fc0 = nn.Linear(8 * 5 * 10, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * 5 * 10)
        x = F.relu(self.fc0(x))
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, y


class EHD4(nn.Module):
    def __init__(self):
        super(EHD4, self).__init__()
        self.fc0 = nn.Linear(840, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 840)
        x = F.relu(self.fc0(x))
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, y


class EHD4_hw(nn.Module):
    def __init__(self):
        super(EHD4_hw, self).__init__()
        self.fc0 = nn.Linear(840, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 840)
        x = F.relu(self.fc0(x))
        return x, 0


class LG4(nn.Module):
    def __init__(self):
        super(LG4, self).__init__()
        self.fc0 = nn.Linear(144, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 144)
        x = F.relu(self.fc0(x))
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, y


class Slice_Fusion0(nn.Module):
    def __init__(self):
        super(Slice_Fusion0, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, (8, 16))
        self.conv2 = nn.Conv2d(8, 16, (4, 6))
        self.fc0 = nn.Linear(16 * 5 * 10, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 5 * 10)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class Slice_Fusion1(nn.Module):
    def __init__(self):
        super(Slice_Fusion1, self).__init__()
        self.convc1 = nn.Conv2d(1, 4, (8, 16))
        self.convc2 = nn.Conv2d(4, 8, (4, 6))
        self.convd1 = nn.Conv2d(1, 4, (8, 16))
        self.convd2 = nn.Conv2d(4, 8, (4, 6))
        self.fc0 = nn.Linear(16 * 5 * 10, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        ct = x[:, 0, :, :, :]
        ct = ct.squeeze(2)
        dt = x[:, 1, :, :, :]
        dt = dt.squeeze(2)
        del x
        ct = F.relu(self.convc1(ct))
        ct = F.relu(self.convc2(ct))
        ct = ct.view(-1, 8 * 5 * 10)
        dt = F.relu(self.convd1(dt))
        dt = F.relu(self.convd2(dt))
        dt = dt.view(-1, 8 * 5 * 10)
        x = torch.cat([ct, dt], dim=1)
        del ct, dt
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class Slice_Fusion2(nn.Module):
    def __init__(self):
        super(Slice_Fusion2, self).__init__()
        self.convc1 = nn.Conv2d(1, 4, (8, 16))
        self.convc2 = nn.Conv2d(4, 8, (4, 6))
        self.convd1 = nn.Conv2d(1, 4, (8, 16))
        self.convd2 = nn.Conv2d(4, 8, (4, 6))
        self.fcc0 = nn.Linear(8 * 5 * 10, 128)
        self.fcd0 = nn.Linear(8 * 5 * 10, 128)
        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        ct = x[:, 0, :, :, :]
        ct = ct.squeeze(2)
        dt = x[:, 1, :, :, :]
        dt = dt.squeeze(2)
        del x
        ct = F.relu(self.convc1(ct))
        ct = F.relu(self.convc2(ct))
        ct = ct.view(-1, 8 * 5 * 10)
        dt = F.relu(self.convd1(dt))
        dt = F.relu(self.convd2(dt))
        dt = dt.view(-1, 8 * 5 * 10)
        ct = F.relu(self.fcc0(ct))
        dt = F.relu(self.fcd0(dt))
        x = torch.cat([ct, dt], dim=1)
        del ct, dt
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class Slice_Fusion3(nn.Module):
    def __init__(self):
        super(Slice_Fusion3, self).__init__()
        self.convc1 = nn.Conv2d(1, 4, (8, 16))
        self.convc2 = nn.Conv2d(4, 8, (4, 6))
        self.convd1 = nn.Conv2d(1, 4, (8, 16))
        self.convd2 = nn.Conv2d(4, 8, (4, 6))
        self.fcc0 = nn.Linear(8 * 5 * 10, 128)
        self.fcd0 = nn.Linear(8 * 5 * 10, 128)
        self.fcc1 = nn.Linear(128, 32)
        self.fcd1 = nn.Linear(128, 32)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        ct = x[:, 0, :, :, :]
        ct = ct.squeeze(2)
        dt = x[:, 1, :, :, :]
        dt = dt.squeeze(2)
        del x
        ct = F.relu(self.convc1(ct))
        ct = F.relu(self.convc2(ct))
        ct = ct.view(-1, 8 * 5 * 10)
        dt = F.relu(self.convd1(dt))
        dt = F.relu(self.convd2(dt))
        dt = dt.view(-1, 8 * 5 * 10)
        ct = F.relu(self.fcc0(ct))
        dt = F.relu(self.fcd0(dt))
        ct = F.relu(self.fcc1(ct))
        dt = F.relu(self.fcd1(dt))
        x = torch.cat([ct, dt], dim=1)
        del ct, dt
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class EL1(nn.Module):
    def __init__(self):
        super(EL1, self).__init__()
        self.fc0 = nn.Linear(984, 128)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 984)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class EL2(nn.Module):
    def __init__(self):
        super(EL2, self).__init__()
        self.fce0 = nn.Linear(840, 128)
        self.fcl0 = nn.Linear(144, 128)
        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        ehd = x[:, :840]
        lg = x[:, 840:]
        del x
        ehd = F.relu(self.fce0(ehd))
        lg = F.relu(self.fcl0(lg))
        x = torch.cat([ehd, lg], dim=1)
        del ehd, lg
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class EL3(nn.Module):
    def __init__(self):
        super(EL3, self).__init__()
        self.fce0 = nn.Linear(840, 128)
        self.fcl0 = nn.Linear(144, 128)
        self.fce1 = nn.Linear(128, 32)
        self.fcl1 = nn.Linear(128, 32)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        ehd = x[:, :840]
        lg = x[:, 840:]
        del x
        ehd = F.relu(self.fce0(ehd))
        lg = F.relu(self.fcl0(lg))
        ehd = F.relu(self.fce1(ehd))
        lg = F.relu(self.fcl1(lg))
        x = torch.cat([ehd, lg], dim=1)
        del ehd, lg
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, 0


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (4, 50), (1, 10))  # in: 1x1x15x320  out 1x6x12x28
        self.pool = nn.MaxPool2d(2, 2)  # out: 1x6x6x14
        self.pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.conv2 = nn.Conv2d(6, 16, (2, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 10)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (4, 50), (1, 10))
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.conv2 = nn.Conv2d(6, 16, (2, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        # import pdb; pdb.set_trace()                             #x: 1x1x15x15x460
        x = x.unsqueeze(1)  # new gen
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net2_short(nn.Module):
    def __init__(self):
        super(Net2_short, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), stride=(1, 1, 3), padding=(2, 0, 0))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6), padding=(2, 0, 0))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = x[:, :, 5:10, :, :]
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class Net2_for_Fusion(nn.Module):
    def __init__(self):
        super(Net2_for_Fusion, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)

        x = F.relu(self.fc1(x))
        y = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class Net2D_GPR(nn.Module):
    def __init__(self):
        super(Net2D_GPR, self).__init__()
        self.conv1 = nn.Conv2d(15, 60, (5, 8), (1, 3), groups=15)
        self.conv2 = nn.Conv2d(60, 240, (3, 6))
        # self.conv3 = nn.Conv3d(12,24,(3,6))
        # self.conv4 = nn.Conv3d(24,48,(3,6))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc1 = nn.Linear(240 * 9 * 35, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        x = x.squeeze(dim=1)  # x: 1x1x15x15x460
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        # x = self.pool(F.relu(self.conv3(x)))            #x: 1x24x7x7x30,   1x24x7x7x15
        # x = self.pool(F.relu(self.conv4(x)))            #x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 240 * 9 * 35)
        y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class Net2_for_Ensemble(nn.Module):
    def __init__(self):
        super(Net2_for_Ensemble, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):  # x: 1x1x15x15x460
        x = self.pool(F.relu(self.conv1(x)))  # x: 1x6x11x11x151, 1x6x11x11x75
        x = self.pool(F.relu(self.conv2(x)))  # x: 1x12x9x9xx70,  1x12x9x9x35
        x = self.pool(F.relu(self.conv3(x)))  # x: 1x24x7x7x30,   1x24x7x7x15
        x = self.pool(F.relu(self.conv4(x)))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        # y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        y = x
        x = self.fc5(x)
        return x, y


class Net2_Ensemble(nn.Module):
    def __init__(self, no_of_nets):
        super(Net2_Ensemble, self).__init__()
        self.fc1 = nn.Linear(5 * no_of_nets, 5 * no_of_nets)
        self.fc2 = nn.Linear(5 * no_of_nets, 5 * no_of_nets)
        self.fc3 = nn.Linear(5 * no_of_nets, 5 * no_of_nets)
        self.fc4 = nn.Linear(5 * no_of_nets, 24)
        self.fc5 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net2_for_picture(nn.Module):
    def __init__(self):
        super(Net2_for_picture, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.conv2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.conv3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.conv4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x)
        x = self.pool(F.relu(x1))
        x2 = self.conv2(x)
        x = self.pool(F.relu(x2))
        x3 = self.conv3(x)
        x = self.pool(F.relu(x3))
        x4 = self.conv4(x)
        x = self.pool(F.relu(x4))  # x: 1x48x5x5x10,   1x48x5x5x5
        x = x.view(-1, 48 * 5 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x0, x1, x2, x3, x4


class Net3_for_Fusion(nn.Module):
    def __init__(self):
        super(Net3_for_Fusion, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, (5, 5, 31))
        self.conv2 = nn.Conv3d(16, 32, (3, 3, 31))
        # self.conv3 = nn.Conv3d(12,24,(3,3,6))
        # self.conv4 = nn.Conv3d(24,48,(3,3,6))
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 4))
        self.fc1 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 48 * 5 * 5 * 5)
        y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class RNN0(nn.Module):
    def __init__(self):
        super(RNN0, self).__init__()
        self.hidden_dim = 200
        self.layer_dim = 5
        self.rnn = nn.RNN(320, 200, 5, batch_first=True, nonlinearity='tanh')
        self.fc0 = nn.Linear(200, 100)
        self.fc1 = nn.Linear(100, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        x, hn = self.rnn(x, h0)
        x = F.relu(self.fc0(x[:, -1, :]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GPRNNDT(nn.Module):
    def __init__(self):
        super(GPRNNDT, self).__init__()
        self.hidden_dim = 200
        self.layer_dim = 5
        self.rnn = nn.RNN(400, 200, 5, batch_first=True, nonlinearity='relu')
        self.fc0 = nn.Linear(200, 100)
        self.fc1 = nn.Linear(100, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 2)
        self.conv1 = nn.Conv2d(1, 6, (5, 8), (1, 3))
        self.conv2 = nn.Conv2d(6, 12, (3, 6))
        self.conv3 = nn.Conv2d(12, 24, (3, 6))
        self.conv4 = nn.Conv2d(24, 48, (3, 6))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.pre = nn.Linear(48 * 5 * 5, 400)

    def forward(self, data):
        x = Variable(torch.zeros(data.size(0), 15, 400)).cuda()
        for dt in range(15):
            z = data[:, :, dt, :, :]
            z = self.pool(F.relu(self.conv1(z)))
            z = self.pool(F.relu(self.conv2(z)))
            z = self.pool(F.relu(self.conv3(z)))
            z = self.pool(F.relu(self.conv4(z)))
            z = z.view(-1, 48 * 5 * 5)
            z = F.relu(self.pre(z))
            z = z.unsqueeze(dim=1)
            x[:, dt, :] = z
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        x, hn = self.rnn(x, h0)
        x = F.relu(self.fc0(x[:, -1, :]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, 0


class multi_CT(nn.Module):
    def __init__(self):
        super(multi_CT, self).__init__()
        self.hidden_dim = 200
        self.layer_dim = 5
        self.rnn = nn.RNN(400, 200, 5, batch_first=True, nonlinearity='relu')
        self.fc0 = nn.Linear(200, 100)
        self.fc1 = nn.Linear(100, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 2)
        self.conv1 = nn.Conv2d(1, 6, (5, 8), (1, 3))
        self.conv2 = nn.Conv2d(6, 12, (3, 6))
        self.conv3 = nn.Conv2d(12, 24, (3, 6))
        self.conv4 = nn.Conv2d(24, 48, (3, 6))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.pre = nn.Linear(48 * 5 * 5, 400)

    def forward(self, data):
        data = data[:, :, 5:10, :, :]
        x = Variable(torch.zeros(data.size(0), 5, 400)).cuda()
        for dt in range(5):
            z = data[:, :, dt, :, :]
            z = self.pool(F.relu(self.conv1(z)))
            z = self.pool(F.relu(self.conv2(z)))
            z = self.pool(F.relu(self.conv3(z)))
            z = self.pool(F.relu(self.conv4(z)))
            z = z.view(-1, 48 * 5 * 5)
            z = F.relu(self.pre(z))
            z = z.unsqueeze(dim=1)
            x[:, dt, :] = z
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).cuda()
        x, hn = self.rnn(x, h0)
        x = F.relu(self.fc0(x[:, -1, :]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, 0


class simple_EHD(nn.Module):
    def __init__(self):
        super(simple_EHD, self).__init__()
        self.fc1 = nn.Linear(840, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 840)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, 0


# class selu_EHD(nn.Module):
#    def __init__(self):
#        super(selu_EHD,self).__init__()
#        self.fc1 = nn.Linear(840,100)
#        self.fc2 = nn.Linear(100,100)
#        self.fc3 = nn.Linear(100,100)
#        self.fc4 = nn.Linear(100,2)
#    def forward(self,x):
#        x = x.view(-1,840)  
#        x = SELU(self.fc1(x))
#        x = SELU(self.fc2(x))
#        x = SELU(self.fc3(x))
#        x = self.fc4(x)
#        return x,0

class LG(nn.Module):
    def __init__(self):
        super(LG, self).__init__()
        # self.fc0 = nn.Linear(840,400)
        self.fc1 = nn.Linear(144, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 144)
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class LG_All(nn.Module):
    def __init__(self):
        super(LG_All, self).__init__()
        # self.fc0 = nn.Linear(840,400)
        self.fc1 = nn.Linear(144 * 15, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 144 * 15)
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, 0


class LG_simplest(nn.Module):
    def __init__(self):
        super(LG_simplest, self).__init__()
        self.fc1 = nn.Linear(144, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x


class LG_supersimple(nn.Module):
    def __init__(self):
        super(LG_supersimple, self).__init__()
        self.fc1 = nn.Linear(144, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.view(-1, 144)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x, 0


class simple_EHD2(nn.Module):
    def __init__(self):
        super(simple_EHD2, self).__init__()
        self.fc0 = nn.Linear(840, 400)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 840)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, 0


class Net2_EHD(nn.Module):
    def __init__(self):
        super(Net2_EHD, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, (5, 2, 4))
        self.conv2 = nn.Conv3d(16, 32, (5, 2, 4))
        self.conv3 = nn.Conv3d(32, 64, (4, 2, 4))
        self.fc1 = nn.Linear(64 * 3 * 2 * 3, 300)
        self.fc2 = nn.Linear(300, 75)
        self.fc3 = nn.Linear(75, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 2 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, 0


class Net_slice(nn.Module):
    def __init__(self):
        super(Net_slice, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(16, 32, (4, 5))
        # self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc1 = nn.Linear(32 * 9 * 9, 100)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


############################    FuserNet    ##################################

class FuserNet(nn.Module):
    def __init__(self):
        super(FuserNet, self).__init__()

        # GPR layers
        self.gpr1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.gpr2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.gpr3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.gpr4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.gpr5 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.gpr6 = nn.Linear(300, 128)
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))

        # slice layers
        self.slice1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.slice2 = nn.Conv2d(12, 24, (4, 5))
        self.slice3 = nn.Conv2d(24, 48, 4)
        self.slice4 = nn.Linear(48 * 6 * 6, 256)
        self.slice5 = nn.Linear(256, 128)

        # fusion layers
        self.fusion1 = nn.Linear(256, 128)
        self.fusion2 = nn.Linear(128, 64)
        self.fusion3 = nn.Linear(64, 16)
        self.fusion4 = nn.Linear(16, 2)

    def forward(self, x):
        gpr = x[:, :15 * 15 * 460]
        gpr = gpr.resize(gpr.size()[0], 15, 15, 460)
        gpr = gpr.unsqueeze(1)
        sli = x[:, 15 * 15 * 460:]
        sli = sli.resize(sli.size()[0], 15, 30)
        sli = sli.unsqueeze(1)
        del x
        gpr = self.pool(F.relu(self.gpr1(gpr)))
        gpr = self.pool(F.relu(self.gpr2(gpr)))
        gpr = self.pool(F.relu(self.gpr3(gpr)))
        gpr = self.pool(F.relu(self.gpr4(gpr)))
        gpr = gpr.view(-1, 48 * 5 * 5 * 5)
        gpr = F.relu(self.gpr5(gpr))
        gpr = F.relu(self.gpr6(gpr))

        sli = F.relu(self.slice1(sli))
        sli = F.relu(self.slice2(sli))
        sli = F.relu(self.slice3(sli))
        sli = sli.view(-1, 48 * 6 * 6)
        sli = F.relu(self.slice4(sli))
        sli = F.relu(self.slice5(sli))

        x = (torch.cat([gpr, sli], dim=1))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = F.relu(self.fusion3(x))
        x = self.fusion4(x)

        return x, 0


############################    FuserNet    ##################################        


############################    FuserNet2    ##################################

class FuserNet2(nn.Module):
    def __init__(self):
        super(FuserNet2, self).__init__()

        # GPR layers
        self.gpr1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.gpr2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.gpr3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.gpr4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.gpr5 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.gpr6 = nn.Linear(300, 128)
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))

        # slice layers
        self.slice1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.slice2 = nn.Conv2d(12, 24, (4, 5))
        self.slice3 = nn.Conv2d(24, 48, 4)
        self.slice4 = nn.Linear(48 * 6 * 6, 256)
        self.slice5 = nn.Linear(256, 128)

        # prefusion layers
        self.prefusion1 = nn.Linear(128, 64)
        self.prefusion2 = nn.Linear(64, 32)
        # fusion layers

        self.fusion1 = nn.Linear(64, 16)
        self.fusion2 = nn.Linear(16, 2)

    def forward(self, x):
        gpr = x[:, :15 * 15 * 460]
        gpr = gpr.resize(gpr.size()[0], 15, 15, 460)
        gpr = gpr.unsqueeze(1)
        sli = x[:, 15 * 15 * 460:]
        sli = sli.resize(sli.size()[0], 15, 30)
        sli = sli.unsqueeze(1)
        del x
        gpr = self.pool(F.relu(self.gpr1(gpr)))
        gpr = self.pool(F.relu(self.gpr2(gpr)))
        gpr = self.pool(F.relu(self.gpr3(gpr)))
        gpr = self.pool(F.relu(self.gpr4(gpr)))
        gpr = gpr.view(-1, 48 * 5 * 5 * 5)
        gpr = F.relu(self.gpr5(gpr))
        gpr = F.relu(self.gpr6(gpr))
        gpr = F.relu(self.prefusion1(gpr))
        gpr = F.relu(self.prefusion2(gpr))

        sli = F.relu(self.slice1(sli))
        sli = F.relu(self.slice2(sli))
        sli = F.relu(self.slice3(sli))
        sli = sli.view(-1, 48 * 6 * 6)
        sli = F.relu(self.slice4(sli))
        sli = F.relu(self.slice5(sli))
        sli = F.relu(self.prefusion1(sli))
        sli = F.relu(self.prefusion2(sli))

        x = (torch.cat([gpr, sli], dim=1))
        x = F.relu(self.fusion1(x))
        x = self.fusion2(x)

        return x, 0


############################    FuserNet2    ################################## 


class GEL(nn.Module):
    def __init__(self):
        super(GEL, self).__init__()

        # GPR layers
        self.gpr1 = nn.Conv3d(1, 6, (5, 5, 8), (1, 1, 3))
        self.gpr2 = nn.Conv3d(6, 12, (3, 3, 6))
        self.gpr3 = nn.Conv3d(12, 24, (3, 3, 6))
        self.gpr4 = nn.Conv3d(24, 48, (3, 3, 6))
        self.gpr5 = nn.Linear(48 * 5 * 5 * 5, 300)
        self.gpr6 = nn.Linear(300, 128)
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2))

        # EHD layers
        self.ehd1 = nn.Linear(840, 400)
        self.ehd2 = nn.Linear(400, 100)
        self.ehd3 = nn.Linear(100, 100)
        self.ehd4 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # LG layers
        self.lg1 = nn.Linear(144, 100)
        self.lg2 = nn.Linear(100, 100)
        self.lg3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # fusion layers
        self.fusion1 = nn.Linear(328, 100)
        self.fusion2 = nn.Linear(100, 100)
        self.fusion3 = nn.Linear(100, 2)

    def forward(self, x):
        gpr = x[:, :15 * 15 * 460]
        gpr = gpr.resize(gpr.size()[0], 1, 15, 15, 460)
        ehd = x[:, 15 * 15 * 460:-144]
        # ehd = ehd.resize(ehd.size()[0],1,840)
        # ehd = ehd.view(-1,840)
        lg = x[:, -144:]
        # lg.view(-1,144)
        del x

        gpr = self.pool(F.relu(self.gpr1(gpr)))
        gpr = self.pool(F.relu(self.gpr2(gpr)))
        gpr = self.pool(F.relu(self.gpr3(gpr)))
        gpr = self.pool(F.relu(self.gpr4(gpr)))
        gpr = gpr.view(-1, 48 * 5 * 5 * 5)
        gpr = F.relu(self.gpr5(gpr))
        gpr = F.relu(self.gpr6(gpr))
        lg = F.relu(self.lg1(lg))
        lg = F.relu(self.lg2(lg))
        lg = F.relu(self.lg3(lg))
        ehd = F.relu(self.ehd1(ehd))
        ehd = F.relu(self.ehd2(ehd))
        ehd = F.relu(self.ehd3(ehd))
        ehd = F.relu(self.ehd4(ehd))

        x = (torch.cat([gpr, ehd, lg], dim=1))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)

        return x, 0


class EL_deep(nn.Module):
    def __init__(self):
        super(EL_deep, self).__init__()

        # surface layers
        self.ehd1 = nn.Linear(840, 400)
        self.ehd2 = nn.Linear(400, 100)
        self.lg1 = nn.Linear(144, 100)
        self.lg2 = nn.Linear(100, 100)

        # deep layers
        self.ehd3 = nn.Linear(100, 64)
        self.ehd4 = nn.Linear(64, 32)
        self.lg3 = nn.Linear(100, 64)
        self.lg4 = nn.Linear(64, 32)

        # surface fusion layers
        self.fusion1 = nn.Linear(200, 100)
        self.fusion2 = nn.Linear(100, 64)

        # deep layers fusion
        self.fusiony1 = nn.Linear(64, 32)

        # second fusion layers
        self.fusionxy1 = nn.Linear(96, 64)
        self.fusionxy2 = nn.Linear(64, 32)
        self.fusionxy3 = nn.Linear(32, 2)

    def forward(self, x):
        ehd = x[:, :-144]
        lg = x[:, -144:]
        del x

        lg = F.relu(self.lg1(lg))
        lg = F.relu(self.lg2(lg))
        ehd = F.relu(self.ehd1(ehd))
        ehd = F.relu(self.ehd2(ehd))

        ehd2 = F.relu(self.ehd3(ehd))
        ehd2 = F.relu(self.ehd4(ehd2))
        lg2 = F.relu(self.lg3(lg))
        lg2 = F.relu(self.lg4(lg2))

        x = (torch.cat([ehd, lg], dim=1))
        y = torch.cat([ehd2, lg2], dim=1)
        del ehd, ehd2, lg, lg2

        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))

        y = F.relu(self.fusiony1(y))

        x = torch.cat([x, y], dim=1)
        del y

        x = F.relu(self.fusionxy1(x))
        x = F.relu(self.fusionxy2(x))
        x = self.fusionxy3(x)

        return x, 0


class EL(nn.Module):
    def __init__(self):
        super(EL, self).__init__()

        # EHD layers
        self.ehd1 = nn.Linear(840, 400)
        self.ehd2 = nn.Linear(400, 100)
        self.ehd3 = nn.Linear(100, 100)
        self.ehd4 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # LG layers
        self.lg1 = nn.Linear(144, 100)
        self.lg2 = nn.Linear(100, 100)
        self.lg3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # fusion layers
        self.fusion1 = nn.Linear(200, 100)
        self.fusion2 = nn.Linear(100, 100)
        self.fusion3 = nn.Linear(100, 2)

    def forward(self, x):
        ehd = x[:, :-144]
        # ehd = ehd.resize(ehd.size()[0],1,840)
        # ehd = ehd.view(-1,840)
        lg = x[:, -144:]
        # lg.view(-1,144)
        del x

        lg = F.relu(self.lg1(lg))
        lg = F.relu(self.lg2(lg))
        lg = F.relu(self.lg3(lg))
        ehd = F.relu(self.ehd1(ehd))
        ehd = F.relu(self.ehd2(ehd))
        ehd = F.relu(self.ehd3(ehd))
        ehd = F.relu(self.ehd4(ehd))

        x = (torch.cat([ehd, lg], dim=1))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)

        return x, 0


class EL_simple(nn.Module):
    def __init__(self):
        super(EL_simple, self).__init__()

        # fusion layers
        self.fusion0 = nn.Linear(984, 200)
        self.fusion1 = nn.Linear(200, 100)
        self.fusion2 = nn.Linear(100, 100)
        self.fusion3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fusion0(x))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)

        return x, 0


class EL_simplest(nn.Module):
    def __init__(self):
        super(EL_simplest, self).__init__()

        # fusion layers
        self.fusion0 = nn.Linear(984, 100)
        self.fusion3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fusion0(x))
        x = self.fusion3(x)

        return x, 0


class EL_CT(nn.Module):
    def __init__(self):
        super(EL_CT, self).__init__()

        # EHD layers
        self.ehd1 = nn.Linear(840, 400)
        self.ehd2 = nn.Linear(400, 100)
        self.ehd3 = nn.Linear(100, 100)
        self.ehd4 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # LG layers
        self.lg1 = nn.Linear(144, 100)
        self.lg2 = nn.Linear(100, 100)
        self.lg3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # slice layers
        self.conv1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.ct1 = nn.Linear(48 * 6 * 6, 100)
        # fusion layers
        self.fusion0 = nn.Linear(300, 200)
        self.fusion1 = nn.Linear(200, 100)
        self.fusion2 = nn.Linear(100, 100)
        self.fusion3 = nn.Linear(100, 2)

    def forward(self, x):
        ehd = x[:, :840]
        # ehd = ehd.resize(ehd.size()[0],1,840)
        # ehd = ehd.view(-1,840)
        lg = x[:, 840:984]
        # lg.view(-1,144)
        ct = x[:, 984:]
        ct = ct.resize(x.size()[0], 1, 15, 30)
        del x

        lg = F.relu(self.lg1(lg))
        lg = F.relu(self.lg2(lg))
        lg = F.relu(self.lg3(lg))
        ehd = F.relu(self.ehd1(ehd))
        ehd = F.relu(self.ehd2(ehd))
        ehd = F.relu(self.ehd3(ehd))
        ehd = F.relu(self.ehd4(ehd))
        ct = F.relu(self.conv1(ct))
        ct = F.relu(self.conv2(ct))
        ct = F.relu(self.conv3(ct))
        ct = ct.view(-1, 48 * 6 * 6)
        ct = F.relu(self.ct1(ct))
        x = (torch.cat([ehd, lg, ct], dim=1))
        x = F.relu(self.fusion0(x))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = self.fusion3(x)

        return x, 0


class EL2(nn.Module):
    def __init__(self):
        super(EL2, self).__init__()

        # EHD layers
        self.ehd1 = nn.Linear(840, 400)
        self.ehd2 = nn.Linear(400, 200)
        self.ehd3 = nn.Linear(200, 200)
        self.ehd4 = nn.Linear(200, 100)
        # self.fc4 = nn.Linear(100,2)

        # LG layers
        self.lg1 = nn.Linear(144, 100)
        self.lg2 = nn.Linear(100, 100)
        self.lg3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100,2)

        # fusion layers
        self.fusion1 = nn.Linear(200, 100)
        self.fusion2 = nn.Linear(100, 100)
        self.fusion22 = nn.Linear(100, 64)
        self.fusion23 = nn.Linear(64, 32)
        self.fusion3 = nn.Linear(32, 2)

    def forward(self, x):
        ehd = x[:, :-144]
        # ehd = ehd.resize(ehd.size()[0],1,840)
        # ehd = ehd.view(-1,840)
        lg = x[:, -144:]
        # lg.view(-1,144)
        del x

        lg = F.relu(self.lg1(lg))
        lg = F.relu(self.lg2(lg))
        lg = F.relu(self.lg3(lg))
        ehd = F.relu(self.ehd1(ehd))
        ehd = F.relu(self.ehd2(ehd))
        ehd = F.relu(self.ehd3(ehd))
        ehd = F.relu(self.ehd4(ehd))

        x = (torch.cat([ehd, lg], dim=1))
        x = F.relu(self.fusion1(x))
        x = F.relu(self.fusion2(x))
        x = F.relu(self.fusion22(x))
        x = F.relu(self.fusion23(x))
        x = self.fusion3(x)

        return x, 0


class Slice_for_Fusion(nn.Module):
    def __init__(self):
        super(Slice_for_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc1 = nn.Linear(48 * 6 * 6, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 48 * 6 * 6)
        y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (8, 16))
        self.conv2 = nn.Conv2d(4, 8, (4, 6))
        self.fc1 = nn.Linear(8 * 5 * 10, 100)
        self.fc5 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * 5 * 10)
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc5(x)
        return x, y


class Slice_Fusion(nn.Module):
    def __init__(self):
        super(Slice_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, (8, 16))
        self.conv2 = nn.Conv2d(8, 16, (4, 6))
        self.fc1 = nn.Linear(16 * 5 * 10, 100)
        self.fc5 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 5 * 10)
        y = x
        x = F.relu(self.fc1(x))
        x = self.fc5(x)
        return x, y


class Slice_DTCT(nn.Module):
    def __init__(self):
        super(Slice_DTCT, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.conv01 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv02 = nn.Conv2d(12, 24, (4, 5))
        self.conv03 = nn.Conv2d(24, 48, 4)
        self.fc01 = nn.Linear(2 * 48 * 6 * 6, 128 * 6)
        self.fc01x = nn.Linear(48 * 6 * 6, 256)
        self.fc02 = nn.Linear(128 * 6, 128 * 3)
        self.fc1 = nn.Linear(128 * 3, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        dt = x[:, 0, :, :, :]
        ct = x[:, 1, :, :, :]
        del x
        dt = F.relu(self.conv1(dt))
        dt = F.relu(self.conv2(dt))
        dt = F.relu(self.conv3(dt))
        dt = dt.view(-1, 48 * 6 * 6)
        ct = F.relu(self.conv01(ct))
        ct = F.relu(self.conv02(ct))
        ct = F.relu(self.conv03(ct))
        ct = ct.view(-1, 48 * 6 * 6)
        y = 0
        x = (torch.cat([dt, ct], dim=1))
        x = F.relu(self.fc01(x))
        x = F.relu(self.fc02(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x, y


class DTCT_simple(nn.Module):
    def __init__(self):
        super(DTCT_simple, self).__init__()
        self.conv1 = nn.Conv2d(2, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc0 = nn.Linear(48 * 6 * 6, 128 * 6)
        self.fc1 = nn.Linear(128 * 6, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 48 * 6 * 6)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, 0


class Slice_for_picture(nn.Module):
    def __init__(self):
        super(Slice_for_picture, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc1 = nn.Linear(48 * 6 * 6, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.squeeze(2)
        x0 = x
        x1 = self.conv1(x)
        x = F.relu(x1)
        x2 = self.conv2(x)
        x = F.relu(x2)
        x3 = self.conv3(x)
        x = F.relu(x3)
        x = x.view(-1, 48 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x0, x1, x2, x3


class Fusion_Net2_Slice(nn.Module):
    def __init__(self):
        super(Fusion_Net2_Slice, self).__init__()
        # self.fc0 = nn.Linear(2 * 48 * 6 * 6, 48 * 6 * 6)
        self.fc1 = nn.Linear(48 * 125 + 48 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Fusion_TWO_Slice(nn.Module):
    def __init__(self):
        super(Fusion_TWO_Slice, self).__init__()
        # self.fc0 = nn.Linear(2 * 48 * 6 * 6, 48 * 6 * 6)
        self.fc1 = nn.Linear(2 * 48 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Fusion_slice(nn.Module):
    def __init__(self):
        super(Fusion_slice, self).__init__()
        # self.fc0 = nn.Linear(2 * 48 * 6 * 6, 48 * 6 * 6)
        self.fc1 = nn.Linear(2 * 48 * 6 * 6, 256)
        # self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x):
        # x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net_slice2(nn.Module):
    def __init__(self):
        super(Net_slice2, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc1 = nn.Linear(2 * 48 * 6 * 6, 256)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(-1, 48 * 6 * 6)
        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv3(x2))
        x2 = x2.view(-1, 48 * 6 * 6)
        x = torch.cat([x1, x2])
        x = x.view(-1, 2 * 48 * 6 * 6)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net_slice3(nn.Module):
    def __init__(self):
        super(Net_slice3, self).__init__()
        self.conv1 = nn.Conv2d(2, 12, (4, 6), (1, 2))
        self.conv2 = nn.Conv2d(12, 24, (4, 5))
        # self.conv3 = nn.Conv2d(24, 48, 4)
        self.fc1 = nn.Linear(24 * 9 * 9, 256)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 2)

    def forward(self, x1):
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        # x1 = F.relu(self.conv3(x1))
        x1 = x1.view(-1, 24 * 9 * 9)
        x = x1
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ZIP_for_Fusion(nn.Module):
    def __init__(self):
        super(ZIP_for_Fusion, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, (6, 4))
        self.conv2 = nn.Conv2d(12, 24, (5, 4))
        self.fc1 = nn.Linear(24 * 8 * 5, 100)
        self.fc3 = nn.Linear(100, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.resize(x.size()[0], 1, 17, 11)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 24 * 8 * 5)
        y = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x, y


class ZIPline_for_Fusion(nn.Module):
    def __init__(self):
        super(ZIPline_for_Fusion, self).__init__()
        self.fc1 = nn.Linear(187, 100)
        self.fc3 = nn.Linear(100, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.view(-1, 187)
        x = F.relu(self.fc1(x))
        y = x
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x, y
