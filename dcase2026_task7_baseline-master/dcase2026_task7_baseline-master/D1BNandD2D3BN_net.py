'''
Taken and modified PANN's CNN14 model architecture written by Qiuqiang Kong
from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
'''
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])

    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=0):
        x = F.relu_(self.bnF[task](self.conv1(input)))
        x = F.relu_(self.bnS[task](self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect pool_type!')

        return x


class MCnn14(nn.Module):
    """
    BN branch mapping:
      0: D1_BN
      1: D2D3_shared_BN
      2: D2_BN
      3: D3_BN
    """

    D1_BN = 0
    D2D3_BN = 1
    D2_BN = 2
    D3_BN = 3

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, nb_tasks=4):
        super(MCnn14, self).__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window='hann',
            center=True,
            pad_mode='reflect',
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.bn0 = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlock(1, 64, nb_tasks)
        self.conv_block2 = ConvBlock(64, 128, nb_tasks)
        self.conv_block3 = ConvBlock(128, 256, nb_tasks)
        self.conv_block4 = ConvBlock(256, 512, nb_tasks)
        self.conv_block5 = ConvBlock(512, 1024, nb_tasks)
        self.conv_block6 = ConvBlock(1024, 2048, nb_tasks)

        self.fc = nn.Linear(2048, classes_num)

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, task=0):
        feature = self.extract_feature(input, task=task)
        return self.fc(feature)

    def extract_feature(self, input, task=0):
        x = self.spectrogram_extractor(input)
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
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def compute_bn_match_score(self, input, task=0):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)

        sample_mean = torch.mean(x, dim=[2, 3])
        sample_var = torch.var(x, dim=[2, 3], unbiased=False)

        running_mean = self.bn0[task].running_mean.unsqueeze(0)
        running_var = self.bn0[task].running_var.unsqueeze(0)

        mean_diff = torch.mean((sample_mean - running_mean) ** 2, dim=1)
        var_diff = torch.mean((sample_var - running_var) ** 2, dim=1)
        return mean_diff + var_diff

    def hierarchical_task_select(self, input):
        """
        Step-1: compare D1_BN vs D2D3_BN.
        Step-2: for non-D1 samples, compare D2_BN vs D3_BN.
        """
        score_d1 = self.compute_bn_match_score(input, task=self.D1_BN)
        score_d23 = self.compute_bn_match_score(input, task=self.D2D3_BN)
        is_d1 = score_d1 <= score_d23

        score_d2 = self.compute_bn_match_score(input, task=self.D2_BN)
        score_d3 = self.compute_bn_match_score(input, task=self.D3_BN)
        is_d2 = score_d2 <= score_d3

        task_ids = torch.full_like(score_d1, fill_value=self.D3_BN, dtype=torch.long)
        task_ids = torch.where(is_d2, torch.full_like(task_ids, self.D2_BN), task_ids)
        task_ids = torch.where(is_d1, torch.full_like(task_ids, self.D1_BN), task_ids)
        return task_ids

    def forward_hierarchical(self, input):
        task_ids = self.hierarchical_task_select(input)
        logits_list = []
        for sample_idx in range(input.shape[0]):
            task = int(task_ids[sample_idx].item())
            logits_list.append(self.forward(input[sample_idx:sample_idx + 1], task=task))
        logits = torch.cat(logits_list, dim=0)
        return logits, task_ids
