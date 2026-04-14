import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
utils_dir = os.path.join(root_dir, 'utils')

for path in [root_dir, current_dir, utils_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

print("current_dir:", current_dir)
print("root_dir:", root_dir)
print("utils_dir:", utils_dir)
print("sys.path[:5]:", sys.path[:5])

# ===== 第三方库 =====
import pandas as pd
import torch
import numpy as np
import argparse
import copy
import time

import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from tqdm import tqdm

# ===== 项目内模块 =====
import config_task7 as config
from config_task7 import (
    sample_rate, mel_bins, fmin, fmax, window_size, hop_size
)

from zyk_domain_net import *
from datasetfactory_task7 import DILDatasetInc as DILDataset
from utilities import *

timestr = time.strftime("%Y%m%d-%H%M%S")

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
    nb_tasks = len(seen_domains) + 1 # +1 is D1
    correct, total = 0, 0
    output_dict = {}
    output_path = config.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, (inputs, targets, audio_file) in enumerate(loader):
        outputs_uncs = None
        inputs = inputs.float()
        inputs = inputs.to(device)
        for task in range(nb_tasks):
            with torch.no_grad():
                outputs = model(inputs, task)
            outputs = torch.softmax(outputs, dim=1)
            if outputs_uncs is None:
                outputs_uncs = outputs.detach()
            else:
                outputs_uncs = torch.concat([outputs_uncs, outputs.detach()], dim=0)

        epsilon = sys.float_info.min
        entropy = -torch.sum(outputs_uncs * torch.log(outputs_uncs + epsilon), dim=-1)
        # Predicted Task Id
        _, task_id = torch.min(entropy, dim=-1, keepdim=False)

        # Classification
        with torch.no_grad():
            outputs = model(inputs, task_id)
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
            class_label = list(config.dict_class_labels.keys())[list(config.dict_class_labels.values()).index(int(targets.cpu().numpy()))]
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


class Learner():
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num, num_tasks, args):
        super(Learner, self).__init__()

        Model = MCnn14  #eval("Transfer_Cnn14")
        self.model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                           classes_num, num_tasks, use_adapter=args.use_adapter)

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
        teacher_model = None
        if self.cur_task > 0:
            teacher_model = copy.deepcopy(self.model)
            teacher_model.freeze_weight()
            teacher_model.to(device)
            teacher_model.eval()

        self.model.freeze_weight()
        shared_parameters = self.model.get_shared_parameters()
        for param in shared_parameters:
            param.requires_grad = True
        self.model.freeze_old_domain_specific(self.cur_task)
        domain_parameters = self.model.get_domain_specific_parameters(self.cur_task)
        for param in domain_parameters:
            param.requires_grad = True

        print(f"params to be adapted")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        criteria = nn.CrossEntropyLoss(ignore_index=-1)
        shared_optimizer = torch.optim.Adam(shared_parameters,
                                            lr=args.learning_rate * args.shared_lr_ratio,
                                            betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=0., amsgrad=True)
        domain_optimizer = torch.optim.Adam(domain_parameters,
                                            lr=args.learning_rate,
                                            betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=0., amsgrad=True)
        shared_scheduler = optim.lr_scheduler.CosineAnnealingLR(shared_optimizer, T_max=args.epoch, eta_min=0.001)
        domain_scheduler = optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=args.epoch, eta_min=0.001)

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()
            self.model.set_old_domain_bn_eval(self.cur_task)

            sum_loss = 0
            sum_dist_loss = 0
            sum_class_loss = 0
            for batch_idx, (audio, target, _) in enumerate(train_loader):
                shared_optimizer.zero_grad()
                domain_optimizer.zero_grad()
                audio = audio.float()
                target = target.float()
                audio = audio.to(device)
                target = target.to(device)
                target_indices = torch.argmax(target, dim=-1)

                # (1) Domain-specific (current BN/adapter) update signal: CE only.
                logits_domain = self.model(audio, self.cur_task) #cur_task is from 0, D1:0, D2:1, D3:2
                ce_domain = criteria(logits_domain, target_indices)
                ce_domain.backward()
                for p in shared_parameters:
                    p.grad = None

                # (2) Shared update signal: CE + KLD distillation.
                for p in domain_parameters:
                    p.requires_grad = False
                logits_shared = self.model(audio, self.cur_task)
                ce_shared = criteria(logits_shared, target_indices)
                kld_loss = torch.zeros(1, device=device)
                if teacher_model is not None and args.lambda_kld > 0:
                    for old_task in range(self.cur_task):
                        with torch.no_grad():
                            teacher_logits = teacher_model(audio, old_task)
                        student_logits = self.model(audio, old_task)
                        kld_loss += F.kl_div(F.log_softmax(student_logits, dim=1),
                                             F.softmax(teacher_logits, dim=1),
                                             reduction='batchmean')
                    kld_loss = kld_loss / self.cur_task
                loss_shared = ce_shared + args.lambda_kld * kld_loss
                loss_shared.backward()
                for p in domain_parameters:
                    p.requires_grad = True

                sum_loss += loss_shared.item()
                sum_class_loss += ce_shared.item()
                sum_dist_loss += kld_loss.item()

                shared_optimizer.step()
                domain_optimizer.step()
                step += 1

                if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                    print('==>>> epoch: {}, batch index: {}, step: {}, train shared loss: {:.3f}, '
                          'ce loss: {:.3f}, kld loss: {:.3f}'.
                          format(epoch_idx, batch_idx + 1, step,
                                 sum_loss / (batch_idx + 1),
                                 sum_class_loss / (batch_idx + 1),
                                 sum_dist_loss / (batch_idx + 1)))

            shared_scheduler.step()
            domain_scheduler.step()

        if args.save:
            save_path = config.save_resume_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(),
                       os.path.join(save_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth'))

    def load_checkpoint(self, device):
        resume_path = os.path.join(config.save_resume_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth')
        state_dict = torch.load(resume_path, map_location=torch.device(device))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print('Missing keys while loading checkpoint: {}'.format(missing))
        if len(unexpected) > 0:
            print('Unexpected keys while loading checkpoint: {}'.format(unexpected))
        print('model trained on Task D{} is loaded'.format(self.cur_task + 1))

    def incremental_setup(self, train_df, valid_df, seen_domains, batch_size, num_workers, device, args):

        self.cur_task += 1

        if self.cur_task == 0:
            self.load_checkpoint(device)
            self.cur_task += 1 #Skip the domain D1

        print("Starting DIL Task D{}".format(self.cur_task + 1))
        self.model.initialize_domain_from_previous(self.cur_task)

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
        return average_accuracy, accuracy_previous

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
                    classes_num, num_tasks, args)

    for task in range(len(dil_tasks)):
        #print('Training on domain:', dil_tasks[task])
        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]
        #eval_df = df_eval #Domain id is not available


        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        seen_accuracy, acc_list = model.acc_prev(
            seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device
        )
        print('Per-domain accuracy: ', acc_list)
        print('Average Accuracy: ', seen_accuracy)

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
    parser_train.add_argument('--use_adapter', action=argparse.BooleanOptionalAction, default=True)
    parser_train.add_argument('--lambda_kld', type=float, default=1.0)
    parser_train.add_argument('--shared_lr_ratio', type=float, default=0.01)
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')