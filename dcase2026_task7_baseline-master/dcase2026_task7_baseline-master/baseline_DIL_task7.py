import pandas as pd

import os
import sys
import importlib.util

import torch
import torch.nn as nn

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import config_task7 as config

import torch.nn.functional as F

from config_task7 import (sample_rate, mel_bins, fmin, fmax, window_size,
                    hop_size)

import torch.optim as optim


from datasetfactory_task7 import DILDatasetInc as DILDataset
from sklearn import metrics
from tqdm import tqdm
from utilities import *
import time
from torch.utils.data import Dataset

timestr = time.strftime("%Y%m%d-%H%M%S")


def _load_domain_model_class(net_path=None):
    """Load MCnn14 from a configurable net path to avoid path/import issues."""
    default_external_path = '/home/dcase_task7/zyk/dcase2026_task7_baseline/baseline/D1VSnonD1net.py'
    legacy_external_path = '/home/dcase_task7/zyk/dcase2026_task7_baseline/baseline/D1 VS non D1net.py'
    search_paths = [
        net_path,
        os.environ.get('DCASE_DOMAIN_NET_PATH'),
        os.path.join(os.path.dirname(__file__), 'domain_net.py'),
        default_external_path,
        legacy_external_path,
    ]

    for path in search_paths:
        if not path:
            continue
        if os.path.isfile(path):
            module_name = f"domain_net_dynamic_{abs(hash(path))}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'MCnn14'):
                print(f"[Info] Loaded MCnn14 from: {path}")
                return module.MCnn14

    # Fall back to local import style when file loading is unavailable.
    from domain_net import MCnn14 as local_mcnn14
    print("[Info] Loaded MCnn14 from python import: domain_net.MCnn14")
    return local_mcnn14


MCnn14 = _load_domain_model_class()


def _compute_accuracy(model, loader, task, device):

    correct, total = 0, 0
    model.eval()
    correct, total = 0, 0
    output_dict = {}

    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, task)
        outputs = torch.softmax(outputs, dim=1)
        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
        append_to_dict(output_dict, 'clipwise_output',
                       outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())
        #print(total, correct)
    #print(total, correct)
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    #print(class_acc)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def _compute_uncertainity(model, loader, seen_domains, device):
    model.eval()
    correct, total = 0, 0
    output_dict = {}
    output_path = config.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    d1_bn_threshold = 0.05
    d1_entropy_threshold = 1.8

    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float()
        inputs = inputs.to(device)

        # Stage-1: D1 vs non-D1 by BN_D1 matching + entropy
        with torch.no_grad():
            outputs_d1 = model(inputs, task=0)
            outputs_d1 = torch.softmax(outputs_d1, dim=1)
            epsilon = sys.float_info.min
            entropy_d1 = -torch.sum(outputs_d1 * torch.log(outputs_d1 + epsilon), dim=-1)
            bn_score_d1 = model.compute_bn_match_score(inputs, task=0)

        is_d1 = (bn_score_d1 <= d1_bn_threshold) & (entropy_d1 <= d1_entropy_threshold)

        # Stage-2: D2 vs D3 domain branch for non-D1 samples
        if torch.all(is_d1):
            task_id = torch.zeros(inputs.shape[0], dtype=torch.long, device=device)
        elif len(seen_domains) == 1:
            task_id = torch.ones(inputs.shape[0], dtype=torch.long, device=device)
        else:
            with torch.no_grad():
                domain_logits = model.forward_domain(inputs, feature_task=0)
                domain_pred = torch.argmax(domain_logits, dim=-1)  # 0->D2, 1->D3
            task_id = torch.where(is_d1, torch.zeros_like(domain_pred), domain_pred + 1)

        # Final class prediction with selected domain-specific BN branch
        outputs_list = []
        with torch.no_grad():
            for sample_idx in range(inputs.shape[0]):
                outputs_sample = model(inputs[sample_idx:sample_idx + 1], int(task_id[sample_idx].item()))
                outputs_list.append(outputs_sample)
        outputs = torch.cat(outputs_list, dim=0)
        outputs = torch.softmax(outputs, dim=1)
        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
        append_to_dict(output_dict, 'clipwise_output',
                       outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())
        with open(os.path.join(output_path  + 'output_' + timestr + '.txt'), 'a') as f:
            # class_label = list(config.dict_class_labels.keys())[list(config.dict_class_labels.values()).index(int(targets.cpu().numpy()))]
            class_label = list(config.dict_class_labels.keys())[list(config.dict_class_labels.values()).index(int(targets.view(-1)[0].item()))]
            f.write(audio_file[0] + '\t' + class_label + '\n')
        # print(total, correct)
    # print(total, correct)
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    #print('confusion matrix:', cm)
    #print('class-wise accuracy', class_acc)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class BinaryDomainDataset(Dataset):
    """Binary dataset for D2-vs-D3 discriminator training."""

    def __init__(self, df, audio_folder):
        self.audio_ds = DILDataset(df, audio_folder)
        domain_map = {'D2': 0, 'D3': 1}
        self.domain_labels = [domain_map[df.iloc[idx]['domain']] for idx in range(len(df))]

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        audio, _, audio_file = self.audio_ds[idx]
        return audio, self.domain_labels[idx], audio_file


class Learner():
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num, num_tasks):
        super(Learner, self).__init__()

        Model = MCnn14  #eval("Transfer_Cnn14")
        self.model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                           classes_num, num_tasks)

        self.classes_seen = 0
        self.known_classes = 0
        self.cur_task = -1
        self.class_increments = []

    def incremental_train(self, train_loader, val_loader, device, args):

        step = 0
        total = 0
        correct = 0
        check_point = 50
        self.model.to(device)

        self.model.freeze_weight()
        if self.cur_task == 0:
            for name, param in self.model.named_parameters():
                if 'conv1' in name or 'conv2' in name or 'fc' in name:
                    param.requires_grad = True
                if 'bn' in name or 'domain_' in name:
                    if '.{}.weight'.format(self.cur_task) in name or '.{}.bias'.format(self.cur_task) in name:
                        param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate
        elif self.cur_task > 0:
            for name, param in self.model.named_parameters():
                if 'bn' in name or 'domain_' in name:
                    if '.{}.weight'.format(self.cur_task) in name or '.{}.bias'.format(self.cur_task) in name:
                        param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate / 10

        print(f"params to be adapted")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        criteria = nn.CrossEntropyLoss(ignore_index=-1)

        optimizer = torch.optim.Adam(non_frozen_parameters, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0., amsgrad=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.001)

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()

            sum_loss = 0
            sum_dist_loss = 0
            sum_class_loss = 0
            for batch_idx, (audio, target, _) in enumerate(train_loader):
                optimizer.zero_grad()
                audio = audio.float()
                target = target.float()
                audio = audio.to(device)
                target = target.to(device)
                target_indices = torch.argmax(target, dim=-1)

                logits = self.model(audio, self.cur_task) #cur_task is from 0, D1:0, D2:1, D3:2
                loss = criteria(logits, target_indices)

                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1

                if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                    print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}'.
                          format(epoch_idx, batch_idx + 1, step, sum_loss / (batch_idx + 1)))

            scheduler.step()

        if args.save:
            save_path = config.save_resume_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(),
                       os.path.join(save_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth'))

    def load_checkpoint(self, device):
        resume_path = os.path.join(config.save_resume_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth')
        checkpoint = torch.load(resume_path, map_location=torch.device(device))
        incompatible = self.model.load_state_dict(checkpoint, strict=False)

        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

        # Backward compatibility: old checkpoints do not contain the new D2/D3 domain head.
        allowed_missing = {'domain_fc.weight', 'domain_fc.bias'}
        real_missing = [k for k in missing_keys if k not in allowed_missing]
        if real_missing:
            raise RuntimeError(
                f'Checkpoint {resume_path} is missing required keys: {real_missing}'
            )

        if unexpected_keys:
            print(f'[Warn] Unexpected keys in checkpoint: {unexpected_keys}')
        if missing_keys:
            print(f'[Warn] Missing keys in checkpoint (initialized randomly): {missing_keys}')

        print('model trained on Task D{} is loaded'.format(self.cur_task + 1))

    def incremental_setup(self, train_df, valid_df, seen_domains, batch_size, num_workers, device, args):

        self.cur_task += 1

        if self.cur_task == 0:
            self.load_checkpoint(device)
            self.cur_task += 1 #Skip the domain D1

        print("Starting DIL Task D{}".format(self.cur_task + 1))

        dataset_train = DILDataset(train_df, config.audio_folder_DIL)
        dataset_val = DILDataset(valid_df, config.audio_folder_DIL)
        #dataset_test = DILDataset(test_df, config.audio_folder_DIL)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True)

        validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                      num_workers=num_workers, pin_memory=True)


        if args.resume:
            # if i == 0:
            self.load_checkpoint(device)
        else:
            self.incremental_train(train_loader, validate_loader, device, args)

        #self.acc_prev(seen_domains, config.df_DIL_dev, config.df_DIL_eval, batch_size, num_workers, device)

    def acc_prev(self, seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device):
        self.model.to(device)
        self.model.eval()
        num_domains = len(seen_domains)
        domain_dict = {}
        accuracy_previous = []
        for domain in range(num_domains):
            #print('previous domain', seen_domains[domain])
            train_df = df_dev_train[df_dev_train['domain'].isin(seen_domains[domain])]
            valid_df = df_dev_test[df_dev_test['domain'].isin(seen_domains[domain])]
            dataset_val = DILDataset(valid_df, config.audio_folder_DIL)
            validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                          num_workers=num_workers, pin_memory=True)

            #prev_accuracy = _compute_accuracy(self.model, validate_loader, domain, device)
            accuracy = _compute_uncertainity(self.model, validate_loader, seen_domains, device)

            print('seen domain: {} and its accuracy: {}'.format(seen_domains[domain], accuracy))

            accuracy_previous.append(accuracy)

        average_accuracy = np.mean(accuracy_previous).item()
        # return average_accuracy
        return average_accuracy, accuracy_previous

    def train_binary_domain_classifier(self, domain_df, batch_size, num_workers, device, args):
        if domain_df.empty:
            return

        self.model.to(device)
        self.model.freeze_weight()
        for param in self.model.domain_fc.parameters():
            param.requires_grad = True

        domain_dataset = BinaryDomainDataset(domain_df, config.audio_folder_DIL)
        domain_loader = torch.utils.data.DataLoader(
            dataset=domain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.domain_fc.parameters(),
            lr=args.domain_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.,
            amsgrad=True,
        )

        self.model.train()
        for epoch_idx in range(1, args.domain_epoch + 1):
            epoch_loss = 0.0
            for audio, domain_target, _ in domain_loader:
                audio = audio.float().to(device)
                domain_target = domain_target.long().to(device)
                optimizer.zero_grad()
                domain_logits = self.model.forward_domain(audio, feature_task=0)
                loss = criterion(domain_logits, domain_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                f"[Domain D2/D3] epoch {epoch_idx}/{args.domain_epoch}, "
                f"loss: {epoch_loss / max(1, len(domain_loader)):.4f}"
            )
'''
def train(args):
    # Arugments & parameters
    classes_num = config.classes_num_DIL
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    df_dev_train = config.df_DIL_dev_train
    df_dev_test = config.df_DIL_dev_test
    #df_eval =

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = args.num_workers

    dil_tasks_0 = ['D1'] # Model is trained on D1 and its checkpoint is released without data

    dil_task_1 = ['D2']

    dil_task_2 = ['D3']

    dil_tasks = [dil_task_1, dil_task_2]
    print('Tasks:', dil_tasks)

    np.random.seed(1193)

    num_tasks = len(dil_tasks) + 1 # + 1 is the domain D1
    seen_domains = []
    model = Learner(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                    classes_num, num_tasks)

    history_accuracies = {} 
    trained_task_names = [] # 记录学习的顺序，方便后续打印文字

    for task in range(len(dil_tasks)):
        #print('Training on domain:', dil_tasks[task])
        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]
        #eval_df = df_eval #Domain id is not available


        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        # seen_accuracy = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device)
        # 【关键修复 1】正确接收两个返回值：平均准确率 和 各个domain的具体准确率列表
        seen_accuracy, acc_list = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device)

        for i, domain_item in enumerate(seen_domains):
            dom_name = domain_item[0]
            if dom_name not in history_accuracies:
                history_accuracies[dom_name] = [] # 如果是第一次见，初始化一个空列表
            history_accuracies[dom_name].append(acc_list[i]) # 追加当下的准确率
        
        # 【关键修复 2】把数据存进字典！
        current_domain_name = dil_tasks[task][0] # 提取名字，比如 'D2'
        initial_accuracies[current_domain_name] = acc_list[-1] # 记录刚学完的巅峰准确率
        
        # 如果是学最后剩下的那个任务，就把最终所有 domain 的准确率记录下来
        if task == len(dil_tasks) - 1:
            for i, domain in enumerate(seen_domains):
                final_accuracies[domain[0]] = acc_list[i]
        print('Average Accuracy: ', seen_accuracy)
    
    # ---------------------------------------------------------
    # 2. 新增：在所有训练结束后，打印一份不会被覆盖的最终对比报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 增量学习全部完成！最终准确率对比报告：")
    print("="*50)
    
    for domain in seen_domains:
        dom_name = domain[0]
        peak_acc = initial_accuracies[dom_name]
        final_acc = final_accuracies.get(dom_name, 0.0)
        forget_drop = peak_acc - final_acc
        
        print(f"Domain {dom_name}:")
        print(f"  - 刚学完时的准确率 (巅峰): {peak_acc:.2f}%")
        print(f"  - 全部学完后的准确率 (最终): {final_acc:.2f}%")
        print(f"  - 准确率下降幅度 (遗忘):   {forget_drop:.2f}%")
        print("-" * 30)
        
    print(f"🎯 最终全部已见 Domain 的平均准确率: {seen_accuracy:.2f}%")
    print("="*50 + "\n")
'''

def train(args):
    # Arugments & parameters
    classes_num = config.classes_num_DIL
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    df_dev_train = config.df_DIL_dev_train
    df_dev_test = config.df_DIL_dev_test

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = args.num_workers

    dil_tasks_0 = ['D1'] # Model is trained on D1 and its checkpoint is released without data
    dil_task_1 = ['D2']
    dil_task_2 = ['D3']
    # 如果你有更多任务，比如 dil_task_3 = ['D4']，直接加到下面这个列表里即可
    dil_tasks = [dil_task_1, dil_task_2] 
    print('Tasks:', dil_tasks)

    np.random.seed(1193)

    num_tasks = len(dil_tasks) + 1 # + 1 is the domain D1
    seen_domains = []
    model = Learner(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                    classes_num, num_tasks)

    # 【核心修改 1】创建一个字典，用来记录每个 domain 在每一步的时间轴轨迹
    # 格式会变成类似于: {'D2': [85.5, 76.2, 70.1], 'D3': [88.0, 82.5]}
    history_accuracies = {} 
    trained_task_names = [] # 记录学习的顺序，方便后续打印文字

    for task in range(len(dil_tasks)):
        current_train_domain = dil_tasks[task][0] # 当前正在学的 domain 名字，比如 'D2'
        trained_task_names.append(current_train_domain)
        
        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]

        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        
        # 接收当前所有已见 domain 的准确率
        seen_accuracy, acc_list = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device)
        
        # 【核心修改 2】把此时此刻每个 domain 的准确率，追加到它们各自的历史列表里
        for i, domain_item in enumerate(seen_domains):
            dom_name = domain_item[0]
            if dom_name not in history_accuracies:
                history_accuracies[dom_name] = [] # 如果是第一次见，初始化一个空列表
            history_accuracies[dom_name].append(acc_list[i]) # 追加当下的准确率

        print(f'Average Accuracy after {current_train_domain}: {seen_accuracy:.2f}')

    # Train the second-stage domain discriminator: D2 vs D3
    domain_df = df_dev_train[df_dev_train['domain'].isin(['D2', 'D3'])]
    model.train_binary_domain_classifier(domain_df, batch_size, num_workers, device, args)
    
    # ---------------------------------------------------------
    # 【核心修改 3】在最后打印出一份按时间轴排布的详细轨迹报告
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("增量学习全部完成！各 Domain 准确率演变轨迹：")
    print("="*50)
    
    for dom_name, acc_history in history_accuracies.items():
        print(f"🎯 Domain {dom_name} 的准确率变化:")
        
        # 找到这个 domain 是在第几个任务学的
        start_index = trained_task_names.index(dom_name) 
        
        for step, acc in enumerate(acc_history):
            # 匹配当前准确率对应的学习阶段
            stage_name = trained_task_names[start_index + step] 
            
            if step == 0:
                print(f"  - 学完 {stage_name} 后: {acc:.2f}% (巅峰值)")
            else:
                drop = acc_history[0] - acc
                print(f"  - 学完 {stage_name} 后: {acc:.2f}% (较巅峰遗忘 {drop:.2f}%)")
                
        print("-" * 30)
        
    print(f"💡 最终全部已见 Domain 的平均准确率: {seen_accuracy:.2f}%")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')

    #parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--num_workers', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--resume', action='store_true', default=False)
    parser_train.add_argument('--save', action='store_true', default=False)
    parser_train.add_argument('--domain_epoch', type=int, default=5)
    parser_train.add_argument('--domain_learning_rate', type=float, default=1e-4)
    parser_train.add_argument('--net_path', type=str, default=None)
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        MCnn14 = _load_domain_model_class(args.net_path)
        train(args)

    else:
        raise Exception('Error argument!')
