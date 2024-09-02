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
from torch.optim.lr_scheduler import MultiStepLR
# import apex
import matplotlib.pyplot as plt
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
    parser.add_argument('--phase', default='test', help='must be train or test') # Set default to test for this script

    parser.add_argument('--assume-yes', default=False, help='Say yes to every prompt')
    parser.add_argument('--mixed', default=False, help='Use mixed precision training')

    parser.add_argument('--config', default='test_config.yaml',help='path to the configuration file')
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
    parser.add_argument('--test-feeder-args', default=dict(), help='the arguments of data loader for test')

    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', type=dict, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--loss', default='CrossEntropy', help='the loss will be used')
    parser.add_argument('--loss_args', default=dict(), help='the arguments of loss')
    

    return parser

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_model()
        self.load_data()

        self.global_step = 0

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
        self.loss = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)

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

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)

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
                score_batches = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                
                for batch_idx, (data, rgb_data, label, verb_label, noun_label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    rgb_data = rgb_data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    verb_label = verb_label.long().cuda(self.output_device)
                    noun_label = noun_label.long().cuda(self.output_device)                    
                    
                    output = self.model(data, rgb_data)
                    if isinstance(output, tuple):
                        output = output[0]

                    loss = self.loss(output, label)

                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

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
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)

            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)

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
        if self.arg.phase == 'test':
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
        with open( main_dir +  p.config, 'r') as f:
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
