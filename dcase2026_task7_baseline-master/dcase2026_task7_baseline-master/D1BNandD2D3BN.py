import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import config_task7 as config
from config_task7 import sample_rate, mel_bins, fmin, fmax, window_size, hop_size

from D1BNandD2D3BN_net import MCnn14
from datasetfactory_task7 import DILDatasetInc as DILDataset
from utilities import append_to_dict, tensor2numpy, get_filename


timestr = time.strftime("%Y%m%d-%H%M%S")


def _compute_uncertainity(model, loader, device):
    model.eval()
    correct, total = 0, 0
    output_dict = {}
    output_path = config.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for inputs, targets, audio_file in loader:
        inputs = inputs.float().to(device)

        with torch.no_grad():
            logits, _ = model.forward_hierarchical(inputs)
            outputs = torch.softmax(logits, dim=1)

        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)

        append_to_dict(output_dict, 'clipwise_output', outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())

        with open(os.path.join(output_path + 'output_' + timestr + '.txt'), 'a') as f:
            class_label = list(config.dict_class_labels.keys())[list(config.dict_class_labels.values()).index(int(targets.view(-1)[0].item()))]
            f.write(audio_file[0] + '\t' + class_label + '\n')

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    cm = metrics.confusion_matrix(
        np.argmax(output_dict['target'], axis=-1),
        np.argmax(output_dict['clipwise_output'], axis=-1),
        labels=None,
    )
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class Learner:
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, num_tasks):
        super(Learner, self).__init__()
        self.model = MCnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, num_tasks)
        self.cur_task = -1

    def _enable_bn_branch_trainable(self, branch_id):
        for name, param in self.model.named_parameters():
            if 'bn' in name and (f'.{branch_id}.weight' in name or f'.{branch_id}.bias' in name):
                param.requires_grad = True

    def incremental_train(self, train_loader, device, args):
        self.model.to(device)
        self.model.freeze_weight()

        # D2 stage (cur_task=1): update D2D3 shared BN + D2 BN
        # D3 stage (cur_task=2): update D2D3 shared BN + D3 BN
        if self.cur_task == 1:
            self._enable_bn_branch_trainable(MCnn14.D2D3_BN)
            self._enable_bn_branch_trainable(MCnn14.D2_BN)
            lr = args.learning_rate / 10
        elif self.cur_task == 2:
            self._enable_bn_branch_trainable(MCnn14.D2D3_BN)
            self._enable_bn_branch_trainable(MCnn14.D3_BN)
            lr = args.learning_rate / 10
        else:
            lr = args.learning_rate

        non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(non_frozen_parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.001)

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()
            sum_loss = 0.0

            for batch_idx, (audio, target, _) in enumerate(train_loader):
                audio = audio.float().to(device)
                target_indices = torch.argmax(target.float().to(device), dim=-1)

                if self.cur_task == 1:
                    # Step-A: update shared D2D3 BN on D2 data
                    optimizer.zero_grad()
                    logits_shared = self.model(audio, task=MCnn14.D2D3_BN)
                    loss_shared = criterion(logits_shared, target_indices)
                    loss_shared.backward()
                    optimizer.step()

                    # Step-B: update D2-specific BN on D2 data
                    optimizer.zero_grad()
                    logits_specific = self.model(audio, task=MCnn14.D2_BN)
                    loss_specific = criterion(logits_specific, target_indices)
                    loss_specific.backward()
                    optimizer.step()

                    loss = loss_shared + loss_specific

                elif self.cur_task == 2:
                    # Step-A: update shared D2D3 BN on D3 data
                    optimizer.zero_grad()
                    logits_shared = self.model(audio, task=MCnn14.D2D3_BN)
                    loss_shared = criterion(logits_shared, target_indices)
                    loss_shared.backward()
                    optimizer.step()

                    # Step-B: update D3-specific BN on D3 data
                    optimizer.zero_grad()
                    logits_specific = self.model(audio, task=MCnn14.D3_BN)
                    loss_specific = criterion(logits_specific, target_indices)
                    loss_specific.backward()
                    optimizer.step()

                    loss = loss_shared + loss_specific
                else:
                    optimizer.zero_grad()
                    logits = self.model(audio, task=MCnn14.D1_BN)
                    loss = criterion(logits, target_indices)
                    loss.backward()
                    optimizer.step()

                sum_loss += loss.item()

            scheduler.step()
            print(f'==>>> epoch: {epoch_idx}, train loss: {sum_loss / max(1, len(train_loader)):.3f}')

        if args.save:
            save_path = config.save_resume_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(), os.path.join(save_path, f'checkpoint_D{self.cur_task + 1}.pth'))

    def load_checkpoint(self, device):
        resume_path = os.path.join(config.save_resume_path, f'checkpoint_D{self.cur_task + 1}.pth')
        self.model.load_state_dict(torch.load(resume_path, map_location=torch.device(device)), strict=False)
        print(f'model trained on Task D{self.cur_task + 1} is loaded')

    def incremental_setup(self, train_df, valid_df, seen_domains, batch_size, num_workers, device, args):
        self.cur_task += 1

        if self.cur_task == 0:
            self.load_checkpoint(device)
            self.cur_task += 1  # skip D1 training data

        print(f"Starting DIL Task D{self.cur_task + 1}")

        dataset_train = DILDataset(train_df, config.audio_folder_DIL)
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        if args.resume:
            self.load_checkpoint(device)
        else:
            self.incremental_train(train_loader, device, args)

    def acc_prev(self, seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device):
        self.model.to(device)
        self.model.eval()

        accuracy_previous = []
        for domain in range(len(seen_domains)):
            valid_df = df_dev_test[df_dev_test['domain'].isin(seen_domains[domain])]
            dataset_val = DILDataset(valid_df, config.audio_folder_DIL)
            validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
            accuracy = _compute_uncertainity(self.model, validate_loader, device)
            print('seen domain: {} and its accuracy: {}'.format(seen_domains[domain], accuracy))
            accuracy_previous.append(accuracy)

        average_accuracy = np.mean(accuracy_previous).item()
        return average_accuracy, accuracy_previous


def train(args):
    classes_num = config.classes_num_DIL
    batch_size = args.batch_size
    df_dev_train = config.df_DIL_dev_train
    df_dev_test = config.df_DIL_dev_test

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    num_workers = args.num_workers

    dil_task_1 = ['D2']
    dil_task_2 = ['D3']
    dil_tasks = [dil_task_1, dil_task_2]
    print('Tasks:', dil_tasks)

    np.random.seed(1193)

    num_tasks = 4  # D1_BN + D2D3_BN + D2_BN + D3_BN
    seen_domains = []
    model = Learner(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, num_tasks)

    history_accuracies = {}
    trained_task_names = []

    for task in range(len(dil_tasks)):
        current_train_domain = dil_tasks[task][0]
        trained_task_names.append(current_train_domain)

        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]

        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        seen_accuracy, acc_list = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device)

        for i, domain_item in enumerate(seen_domains):
            dom_name = domain_item[0]
            if dom_name not in history_accuracies:
                history_accuracies[dom_name] = []
            history_accuracies[dom_name].append(acc_list[i])

        print(f'Average Accuracy after {current_train_domain}: {seen_accuracy:.2f}')

    print("\n" + "=" * 50)
    print("增量学习全部完成！各 Domain 准确率演变轨迹：")
    print("=" * 50)
    for dom_name, acc_history in history_accuracies.items():
        print(f"🎯 Domain {dom_name} 的准确率变化:")
        start_index = trained_task_names.index(dom_name)
        for step, acc in enumerate(acc_history):
            stage_name = trained_task_names[start_index + step]
            if step == 0:
                print(f"  - 学完 {stage_name} 后: {acc:.2f}% (巅峰值)")
            else:
                drop = acc_history[0] - acc
                print(f"  - 学完 {stage_name} 后: {acc:.2f}% (较巅峰遗忘 {drop:.2f}%)")
        print("-" * 30)
    print(f"💡 最终全部已见 Domain 的平均准确率: {seen_accuracy:.2f}%")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='D1BN + D2D3BN + D2BN + D3BN baseline.')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--num_workers', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--resume', action='store_true', default=False)
    parser_train.add_argument('--save', action='store_true', default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')
