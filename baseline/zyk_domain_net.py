'''
Taken and modified  PANN's CNN14 model architecture written by Qiuqiang Kong
from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
'''
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks, use_adapter=False):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(nb_tasks)])
        self.use_adapter = use_adapter
        if self.use_adapter:
            # Domain-specific parallel residual adapter (1x1 conv branch).
            self.adapter = nn.ModuleList([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
                for _ in range(nb_tasks)
            ])
        else:
            self.adapter = None

        #self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1):

        x = input
        x_main = F.relu_(self.bnF[task](self.conv1(x)))
        x_main = self.bnS[task](self.conv2(x_main))
        if self.use_adapter:
            x = F.relu_(x_main + self.adapter[task](input))
        else:
            x = F.relu_(x_main)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class MCnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1, use_adapter=False):

        super(MCnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)
        self.nb_tasks = nb_tasks
        self.use_adapter = use_adapter

        self.bn0 = nn.ModuleList([nn.BatchNorm2d(64) for i in range(nb_tasks)])
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048, nb_tasks=nb_tasks, use_adapter=use_adapter)
        self.conv_blocks = [
            self.conv_block1, self.conv_block2, self.conv_block3,
            self.conv_block4, self.conv_block5, self.conv_block6
        ]

        self.fc = nn.Linear(2048, classes_num)




    def get_output_dim(self):
        return self.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):

        if second_iter:
            in_features = self.fc.in_features
            out_features = self.fc.out_features

            print("in_features:", in_features, "out_features:", out_features)
            new_out_features = new_dim
            num_new_classes = new_dim - out_features
            new_fc = nn.Linear(in_features, out_features + num_new_classes)

            new_fc.weight.data[:out_features] = self.fc.weight.data
            new_fc.bias.data[:out_features] = self.fc.bias.data
            self.fc = new_fc
            self.n_classes = new_out_features

        else:
            in_features = self.fc.in_features
            out_features = self.fc.out_features

            print("in_features:", in_features, "out_features:", out_features)
            new_out_features = new_dim
            num_new_classes = new_dim - out_features
            new_fc = nn.Linear(in_features, out_features + num_new_classes)

            new_fc.weight.data[:out_features] = self.fc.weight.data
            new_fc.bias.data[:out_features] = self.fc.bias.data
            self.fc = new_fc
            self.n_classes = new_out_features

    def freeze_weight_conv(self):
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.conv_block2.parameters():
            param.requires_grad = False
        for param in self.conv_block3.parameters():
            param.requires_grad = False

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_domain_specific_parameters(self, task):
        params = list(self.bn0[task].parameters())
        for block in self.conv_blocks:
            params.extend(list(block.bnF[task].parameters()))
            params.extend(list(block.bnS[task].parameters()))
            if block.use_adapter:
                params.extend(list(block.adapter[task].parameters()))
        return params

    def get_shared_parameters(self):
        shared_params = []
        for name, param in self.named_parameters():
            if '.bn0.' in name or '.bnF.' in name or '.bnS.' in name or '.adapter.' in name:
                continue
            shared_params.append(param)
        return shared_params

    def initialize_domain_from_previous(self, task):
        if task <= 0:
            return
        prev = task - 1
        self.bn0[task].load_state_dict(copy.deepcopy(self.bn0[prev].state_dict()))
        for block in self.conv_blocks:
            block.bnF[task].load_state_dict(copy.deepcopy(block.bnF[prev].state_dict()))
            block.bnS[task].load_state_dict(copy.deepcopy(block.bnS[prev].state_dict()))
            if block.use_adapter:
                block.adapter[task].load_state_dict(copy.deepcopy(block.adapter[prev].state_dict()))

    def freeze_old_domain_specific(self, cur_task):
        for task in range(cur_task):
            for param in self.get_domain_specific_parameters(task):
                param.requires_grad = False

    def set_old_domain_bn_eval(self, cur_task):
        # Keep historical domain BNs fixed during incremental training.
        for task in range(cur_task):
            self.bn0[task].eval()
            for block in self.conv_blocks:
                block.bnF[task].eval()
                block.bnS[task].eval()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                nn.init.xavier_uniform_(m.weight)

                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        m.bias.data.fill_(0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.fill_(0.)
                m.weight.data.fill_(1.)



    def forward(self, input, task=1):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.fc(x)
        return x