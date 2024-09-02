#!/usr/bin/env python
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import count_params, import_class, get_loss_func

def init_seed(seed):
    """
    Initialize seeds for all random modules (CUDA, torch, numpy, random)
    """
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_parser():
    """
    parsing parameters defined from cmd or config file. If not found in either, use the default values    
    """
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='HandFormer')

    parser.add_argument('--work-dir', type=str, default="", help='the work folder for storing results')
    parser.add_argument('--phase', default='train', help='must be train or test')

    parser.add_argument('--optimizer-states', type=str, help='path of previously saved optimizer states')
    parser.add_argument('--checkpoint', type=str, help='path of previously saved training checkpoint')
    parser.add_argument('--assume-yes', default=False, help='Say yes to every prompt')
    parser.add_argument('--mixed', default=False, help='Use mixed precision training')

    parser.add_argument('--config', default='train_config.yaml',help='path to the configuration file')
    parser.add_argument('--save-score', type=str2bool, default=True,help='if ture, the classification score will be stored')
    parser.add_argument('--seed', type=int, default=random.randrange(200), help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--eval-start', type=int, default=1, help='The epoch number to start evaluating models')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')

    parser.add_argument('--feeder', default='feeders.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')

    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', type=dict, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--amp-opt-level', type=int, default=1, help='NVIDIA Apex AMP optimization level')

    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--forward-batch-size', type=int, default=16, help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start or resume training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='regularization with weight decay for optimizer')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--loss', default='CrossEntropy', help='the loss will be used')
    parser.add_argument('--loss_args', default=dict(), help='the arguments of loss')
    parser.add_argument('--warm_up_epoch', type=int, default=5, help='grow learning rate in the first few epochs')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    parser.add_argument('--add_verb_loss', default=False, help='on adding verb loss to the total loss')
    parser.add_argument('--add_noun_loss', default=False, help='on adding noun loss to the total loss')
    parser.add_argument('--add_feat_loss', default=False, help='on adding feature loss to the total loss')

    parser.add_argument('--action_loss_weight', default=1.0, help='loss component weight')
    parser.add_argument('--verb_loss_weight', default=0.0, help='loss component weight')
    parser.add_argument('--noun_loss_weight', default=0.0, help='loss component weight')
    parser.add_argument('--feat_loss_weight', default=0.0, help='loss component weight')

    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Conflict! Abort... Dir not removed:', logdir)
                        exit()

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        if self.arg.mixed:
            # Mixed precision technique.  https://pytorch.org/docs/stable/notes/amp_examples.html
            self.print_log('*************************************')
            self.print_log('*** Using Mixed Precision Training ***')
            self.print_log('*************************************')
            # Creates a GradScaler once at the beginning of training.
            self.scaler = torch.cuda.amp.GradScaler()

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        # Back up model code and this file to the work_dir
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('', __file__), self.arg.work_dir)

        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss   = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)
        self.loss_v = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)
        self.loss_n = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)

        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1]) # weights-10-28920.pt --> 28920
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            weights = torch.load(self.arg.weights)

            """ 
            Store reference to copies of the weight tensors on CUDA memory along with the keys in the dictionary
            """
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            
            """ 
            To ignore specific weights
            """
            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                """ 
                The model's weights did not match with given weights
                Update the ones that are available and keep log of the missing ones
                """
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        self.param_groups = defaultdict(list) # Dictionary, if key does not exist, will return an empty list

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any -- through checkpoint
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    
    # Step LR Decay
    def adjust_learning_rate(self, epoch):
        self.print_log('adjust learning rate, using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam'  or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * ( self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
                    
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train': # Default is train. For testing, explicitly set phase to test to avoid this data loading
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save args to work_dir
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([ [k.split('module.')[-1], v.cpu()]  for k, v in state_dict.items() ])
        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, epoch, save_model=False):
        self.model.train()
        self.adjust_learning_rate(epoch)

        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.6f}')

        process = tqdm(loader, dynamic_ncols=True)


        for batch_idx, (data, rgb_data, label, verb_label, noun_label, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                rgb_data = rgb_data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                verb_label = verb_label.long().cuda(self.output_device)
                noun_label = noun_label.long().cuda(self.output_device)

            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_rgb_data, batch_label, batch_verb_label, batch_noun_label = data[left:right], rgb_data[left:right], \
                                                        label[left:right], verb_label[left:right], noun_label[left:right]

                if self.arg.mixed:                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # forward
                        output = self.model(batch_data, batch_rgb_data)                        
                        if not isinstance(output, tuple):
                            feat_loss = 0 # If not tuple, then no other loss
                        else:
                            if len(output)==2: # Output and feature loss
                                output, feat_loss = output
                                feat_loss = feat_loss.mean()
                            elif len(output)==3: # Output (action), verb and noun
                                output, verb_output, noun_output = output
                                feat_loss = 0
                            elif len(output)==4: # Everything
                                output, verb_output, noun_output, feat_loss = output
                                feat_loss = feat_loss.mean()

                        category_loss = self.loss(output, batch_label)
                        verb_category_loss = self.loss_v(verb_output, batch_verb_label) if self.arg.add_verb_loss else 0
                        noun_category_loss = self.loss_n(noun_output, batch_noun_label) if self.arg.add_noun_loss else 0
                        feat_loss = feat_loss if self.arg.add_feat_loss else 0

                        loss = ((self.arg.action_loss_weight * category_loss \
                                + self.arg.verb_loss_weight * verb_category_loss \
                                + self.arg.noun_loss_weight * noun_category_loss\
                                + self.arg.feat_loss_weight * feat_loss) / splits)    
                        
                    self.scaler.scale(loss).backward()
                    
                else:
                    #### SAME BLOCK AS ABOVE ### 
                    # forward
                    output = self.model(batch_data, batch_rgb_data)
                    if not isinstance(output, tuple): # If not tuple, then no other loss
                            feat_loss = 0
                    else:
                        if len(output)==2: # Output and feature loss
                            output, feat_loss = output
                            feat_loss = feat_loss.mean()
                        elif len(output)==3: # Output (action), verb and noun
                            output, verb_output, noun_output = output
                            feat_loss = 0
                        elif len(output)==4: # Everything
                            output, verb_output, noun_output, feat_loss = output
                            feat_loss = feat_loss.mean()

                    category_loss = self.loss(output, batch_label)
                    verb_category_loss = self.loss_v(verb_output, batch_verb_label) if self.arg.add_verb_loss else 0
                    noun_category_loss = self.loss_n(noun_output, batch_noun_label) if self.arg.add_noun_loss else 0
                    feat_loss = feat_loss if self.arg.add_feat_loss else 0
                    
                    loss = ((self.arg.action_loss_weight * category_loss \
                            + self.arg.verb_loss_weight * verb_category_loss \
                            + self.arg.noun_loss_weight * noun_category_loss \
                            + self.arg.feat_loss_weight * feat_loss) / splits)

                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())

                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step) # Multiply back as this loss was divided by splits
                self.train_writer.add_scalar('category_loss', category_loss.item(), self.global_step) # Primary loss
                if self.arg.add_verb_loss:
                    self.train_writer.add_scalar('verb_category_loss', verb_category_loss.item(), self.global_step)
                if self.arg.add_noun_loss:
                    self.train_writer.add_scalar('noun_category_loss', noun_category_loss.item(), self.global_step)
                if self.arg.add_feat_loss:
                    self.train_writer.add_scalar('feat_loss', feat_loss.item(), self.global_step)

            if self.arg.mixed:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(
            f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                category_loss_values = []
                verb_loss_values = []
                noun_loss_values = []
                feat_loss_values = []

                score_batches = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)

                for batch_idx, (data, rgb_data, label, verb_label, noun_label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    rgb_data = rgb_data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    verb_label = verb_label.long().cuda(self.output_device)
                    noun_label = noun_label.long().cuda(self.output_device)

                    # forward
                    output = self.model(data, rgb_data, label)
                    if not isinstance(output, tuple): # If not tuple, then no other loss
                        feat_loss = 0
                    else:
                        if len(output)==2: # Output and feature loss
                            output, feat_loss = output
                            feat_loss = feat_loss.mean()
                        elif len(output)==3: # Output (action), verb and noun
                            output, verb_output, noun_output = output
                            feat_loss = 0
                        elif len(output)==4: # Everything
                            output, verb_output, noun_output, feat_loss = output
                            feat_loss = feat_loss.mean()

                    category_loss = self.loss(output, label)
                    verb_category_loss = self.loss_v(verb_output, verb_label) if self.arg.add_verb_loss else 0
                    noun_category_loss = self.loss_n(noun_output, noun_label) if self.arg.add_noun_loss else 0
                    feat_loss = feat_loss if self.arg.add_feat_loss else 0

                    loss = (self.arg.action_loss_weight * category_loss \
                                + self.arg.verb_loss_weight * verb_category_loss \
                                + self.arg.noun_loss_weight * noun_category_loss\
                                + self.arg.feat_loss_weight * feat_loss)
                    
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())
                    category_loss_values.append(category_loss.item())
                    if self.arg.add_verb_loss:
                        verb_loss_values.append(verb_category_loss.item())
                    if self.arg.add_noun_loss:
                        noun_loss_values.append(noun_category_loss.item())
                    if self.arg.add_feat_loss:
                        feat_loss_values.append(feat_loss.item())
                    

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_batches)
            mean_loss = np.mean(loss_values)
            mean_category_loss = np.mean(category_loss_values)
            mean_verb_category_loss = np.mean(verb_loss_values) if self.arg.add_verb_loss else 0
            mean_noun_category_loss = np.mean(noun_loss_values) if self.arg.add_noun_loss else 0
            mean_feat_loss = np.mean(feat_loss_values) if self.arg.add_feat_loss else 0

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('loss', mean_loss, self.global_step)
                self.val_writer.add_scalar('category_loss', mean_category_loss, self.global_step)
                if self.arg.add_verb_loss:
                    self.val_writer.add_scalar('verb_category_loss', mean_verb_category_loss, self.global_step)
                if self.arg.add_noun_loss:
                    self.val_writer.add_scalar('noun_category_loss', mean_noun_category_loss, self.global_step)
                if self.arg.add_feat_loss:
                    self.val_writer.add_scalar('feat_loss', mean_feat_loss, self.global_step)
                
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        with torch.cuda.device('cuda:'+str(self.output_device)):
            torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()
    # HandFormer code directory
    # main_dir = "/home/salman/HandFormer/HandFormer/"
    # or get this file's directory
    main_dir = os.path.dirname(os.path.realpath(__file__))  + '/'
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        print(main_dir + p.config)
        with open(main_dir +  p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    with torch.cuda.device('cuda:0'):
        torch.cuda.empty_cache()
    main()
