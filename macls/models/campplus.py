import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


# 创建非线性激活函数
def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


# 统计池化
def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


# 时序卷积层
class TDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, out_channels, reduction=2):
        super(SEBlock, self).__init__()
        self.out_channels = out_channels  # 32
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 计算全局通道权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # 128 → 64
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, out_channels),  # 64 → 32
            nn.Sigmoid()
        )
        self.restore_conv = nn.Conv1d(channels, out_channels, 1)  # 🚀 128 → 32

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1)  # 🚀 计算全局通道权重
        # y.shape = [1, 128]
        y = self.fc(y)  # 🚀 通过 MLP 计算注意力
        # y.shape = [1, 32]
        y = y.unsqueeze(-1)  # 🚀 变成 [1, 32, 1]
        x = self.restore_conv(x)  # 🚀 变成 [1, 32, T]

        return x * y  # 🚀 **广播运算，通道加权**


class CAMLayer(nn.Module):
    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        # print(f'CAMLayer参数: in_channels:{bn_channels},out_channels:{out_channels},kernel_size:{kernel_size},stride:{stride},padding:{padding},dilation:{dilation},bias:{bias}')
        # print('===================================================================================')
        # TDNN
        self.linear_local = nn.Conv1d(bn_channels,
                                      out_channels,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 提取局部特征
        y = self.linear_local(x)
        # 全局平均值，得到 整个序列的平均信息。   计算 局部池化信息，类似 自适应池化。
        # 两者相加，形成 context，用于计算通道注意力
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        print(y.size())
        return y * m

    # 局部池化
    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        # print(f'CAMDenseTDNNLayer参数: in_channels:{in_channels + in_channels} ,out_channels:{out_channels},bn_channels:{bn_channels},kernel_size:{kernel_size},stride:{stride},dilation:{dilation},bias:{bias}')

        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        # self.cam_layer = CAMLayer(bn_channels,
        #                           out_channels,
        #                           kernel_size,
        #                           stride=stride,
        #                           padding=padding,
        #                           dilation=dilation,
        #                           bias=bias)
        self.se_layer = SEBlock(bn_channels, out_channels)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x, use_reentrant=False)
        else:
            x = self.bn_function(x)
        # x = self.cam_layer(self.nonlinear2(x))
        x = self.se_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        # print(f'CAMDenseTDNNBlock参数：num_layers：{num_layers},in_channels:{in_channels},out_channels：{out_channels},bn_channels:{bn_channels},kernel_size:{kernel_size},stride:{stride},dilation:{dilation},bias:{bias}')
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(in_channels=in_channels + i * out_channels,
                                      out_channels=out_channels,
                                      bn_channels=bn_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      dilation=dilation,
                                      bias=bias,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            # print(f"Before layer : {x.shape}")  # 🚨 Debug
            x = torch.cat([x, layer(x)], dim=1)
            # print(f"After layer : {x.shape}")  # 🚨 Debug
            # print("==========================================")
        return x


class TransitLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

# 残差模块
class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=(stride, 1),
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # 判断是否需要 1x1 卷积，如果输入输出前后通道不一致，则需要1x1卷积
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 特征提取模块
class FCM(nn.Module):
    """
    FCM:Feature Combination Module, CNN特征提取模块
    逐步降低时间分辨率，提取高级特征;
    输出一个紧凑的特征矩阵，作为后续 TDNN 层的输入
    """
    def __init__(self,
                 block=BasicResBlock,  # 使用 ResNet 基本块
                 num_blocks=[2, 2],   # 两个 ResNet Block，每个有 2 个 ResNet 层
                 m_channels=32,  # CNN 的通道数
                 feat_dim=80):  # 频率维度（输入特征大小）
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (math.ceil(feat_dim / 8))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一层的步长为stride，后面的都是1 ，[2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    def __init__(self,
                 num_class,
                 input_size,
                 embd_dim=512,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=input_size)  # input_size=80
        channels = self.head.out_channels
        self.embd_dim = embd_dim

        self.xvector = nn.Sequential(
            OrderedDict([('tdnn', TDNNLayer(channels,
                                            init_channels,
                                            5,
                                            stride=2,
                                            dilation=1,
                                            padding=-1,
                                            config_str=config_str)),
                         ]))
        channels = init_channels  # 128
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module('transit%d' % (i + 1),
                                    TransitLayer(channels,
                                                 channels // 2,
                                                 bias=False,
                                                 config_str=config_str))
            channels //= 2

        self.xvector.add_module(
            'out_nonlinear', get_nonlinear(config_str, channels))

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module('dense', DenseLayer(channels * 2, embd_dim, config_str='batchnorm_'))
        # 分类层
        self.fc = nn.Linear(embd_dim, num_class)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T) 交换1、2位置
        x = self.head(x)
        x = self.xvector(x)
        x = self.fc(x)
        return x
