import sys
sys.path.extend(['../'])

import torch
import pickle
import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset
import lmdb
import pandas as pd
import os
import torchvision.models as models

from feeders import tools


# extract the TSM feature for a particular key
def extract_by_key(env, key):
    ''' 
        input:
            env: loaded lmdb environment
            key: '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
                 example: nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
        output: a 2048-D np-array (TSM feature with ResNet50) or 1536-D np-array (DINOv2 feature)
    '''
    with env.begin() as e:
        data = e.get(key.strip().encode('utf-8'))
        if data is None:
            print(f'[ERROR] Key {key} does not exist !!!')
            exit()
        data = np.frombuffer(data, 'float32')  # convert to numpy array
    return data


class Feeder(Dataset):
    def __init__(self, data_path, rgb_data_path, label_path, split_name, target_type='action',
                 sample_cnt_vid=8, sample_cnt_pose=120, sampling_strategy='default', p_interval=[1.0],
                 rgb_feature_source='tsm', crop_scale='full', rgb_sampling_within_window='first', pose_fps=60,
                 debug=False):
        """
        :param data_path: contains processed skeleton/pose data
        :param rgb_data_path: path where the frame lists for clips (path to the jpg) are stored (HAND VID RAW)
        :param label_path: contains sample names and labels
        :param split_name: train / validation / test_with_labels
        :param target_type: action / verb / noun
        :param sample_cnt_vid: number of frames to sample for video
        :param sample_cnt_pose: number of frames to sample for pose
        :param sampling_strategy: default / crop_and_resize.
                                    - crop_and_resize: For pose, crop and resize along temporal dim.
                                    - pick a segment and resize to the desired length
        :param p_interval: Used when sampling strategy is 'crop_and_resize'. Can be either a float (p) or a pair.
                            - if pair, a float (p) is sampled from the interval
                            - p <= 1, crop and keep p% of the video with random offset
        :param rgb_feature_source: tsm / dino / resnet / tsm_ego_e4 / tsm_ego_e3 / none
        :param crop_scale: full / cropped / both. For DINOv2, hand crop and/or full image as input.
        :param rgb_sampling_within_window: first / mid / random. Sampling strategy for RGB frames within the micro-action window.
        :param pose_fps: frame rate of the pose data. From 60 to 15 fps, no decrease in performance was observed.
        :param debug: If true, only use the first 100 samples
        """
        self.data_path = data_path
        self.rgb_data_path = rgb_data_path
        self.label_path = label_path
        self.split_name = split_name
        self.target_type = target_type
        self.sample_cnt_vid = sample_cnt_vid # 8
        self.sample_cnt_pose = sample_cnt_pose # 120
        self.sampling_strategy = sampling_strategy
        self.p_interval = p_interval
        self.rgb_feature_source = rgb_feature_source
        self.crop_scale = crop_scale
        self.rgb_sampling_within_window = rgb_sampling_within_window
        self.pose_fps = pose_fps
        self.debug = debug

        self.transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        
        ### NOTE: Annotations, jpg frames, tsm/dino feat -> 30 fps | pose data stored @ 60 fps

        ### TODO: Update the RGB feature sources here for new servers or machines
        if self.rgb_feature_source!='none': # If none, no need to load RGB features
            if self.crop_scale in ['full', 'both']: # full sized images are used
                if self.rgb_feature_source=='tsm':
                    lmdb_path_fullImg = "/mnt/data/Datasets/Assembly101/TSM_features/C10119_rgb"
                    # Default is view4. Alternative views: C10115_rgb" # C10095_rgb" # C10119_rgb" -- if these are used remember to replace the lmdb key later
                elif self.rgb_feature_source=='tsm_ego_e4':
                    # Taking ego view 4 --> HMC_84358933_mono10bit or HMC_21179183_mono10bit : e4
                    lmdb_path_fullImg = "/mnt/data/Datasets/Assembly101/TSM_features/HMC_84358933_mono10bit"
                    lmdb_path_fullImg_2 = "/mnt/data/Datasets/Assembly101/TSM_features/HMC_21179183_mono10bit"
                elif self.rgb_feature_source=='tsm_ego_e3':
                    # Taking ego view 3 --> HMC_84355350_mono10bit or HMC_21110305_mono10bit : e3
                    lmdb_path_fullImg = "/mnt/data/Datasets/Assembly101/TSM_features/HMC_84355350_mono10bit"
                    lmdb_path_fullImg_2 = "/mnt/data/Datasets/Assembly101/TSM_features/HMC_21110305_mono10bit"

                elif self.rgb_feature_source=='dino':
                    ### Generate the DINOv2 features before running this code. ###
                    lmdb_path_fullImg = "/home/salman/dinov2_feats_assembly/lmdb_fullImg"
                elif self.rgb_feature_source=='resnet':
                    lmdb_path_fullImg = "/mnt/data/salman/assembly101_resnet50_feats/lmdb_fullImg"                

                self.env_fullImg = lmdb.open(lmdb_path_fullImg, readonly = True, lock=False)
                if self.rgb_feature_source=='tsm_ego_e3' or self.rgb_feature_source=='tsm_ego_e4':
                    # Some egocentric videos have an alternate name for the same camera view. So, we need to load two lmdb files.
                    self.env_fullImg_2 = lmdb.open(lmdb_path_fullImg_2, readonly = True, lock=False)

            if self.crop_scale in ['cropped', 'both']:
                if self.rgb_feature_source in ['tsm', 'tsm_ego_e4', 'tsm_ego_e3']:
                    print("Cropped images' TSM features not available! Abort.")
                    sys.exit()
                elif self.rgb_feature_source=='dino':
                    lmdb_path_croppedImg = "/home/salman/dinov2_feats_assembly/lmdb_croppedImg"
                elif self.rgb_feature_source=='resnet':
                    lmdb_path_croppedImg = "/mnt/data/salman/assembly101_resnet50_feats/lmdb_croppedImg_part"

                self.env_croppedImg = lmdb.open(lmdb_path_croppedImg, readonly = True, lock=False)

        self.load_verb_obj_breakdown() # To get verb and noun labels from action label
        self.load_data()

    def load_verb_obj_breakdown(self):
        action_csv_path = '/home/salman/Assembly101_FG_annotations/actions.csv'
        if not os.path.exists(action_csv_path):
            print("actions.csv not found. Correct path in feeder.py load_verb_obj_breakdown()!")

        self.action_breakdown = {} # (verb_id, noun_id) for each action_id
        actdf = pd.read_csv(action_csv_path)
        
        for _, row in actdf.iterrows():
            self.action_breakdown[row['action_id']] = (row['verb_id'], row['noun_id'])

    def load_data(self):
        try:
            with open(self.label_path + self.split_name + '_label.pkl') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path + self.split_name + '_label.pkl', 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        self.verb_label = [self.action_breakdown[lbl][0] for lbl in self.label]
        self.noun_label = [self.action_breakdown[lbl][1] for lbl in self.label]


        if self.debug:
            self.label = self.label[0:1000]
            self.verb_label = self.verb_label[0:1000]
            self.noun_label = self.noun_label[0:1000]
            self.sample_name = self.sample_name[0:1000]
        
    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self
    
    def _sample_indices(self, total_frames, sample_cnt):
        """
        This function can be invoked for pose data sampling during training. Randomness (with different strategies) is applied.
        :param total_frames: For pose, total pose frames (T).
        :param sample_cnt: how many frames we need to sample.
        :return: list
        """
        average_duration = total_frames // sample_cnt
        if average_duration > 0:
            # Sample one random frame from each segment
            # selected_indices = np.multiply(list(range(sample_cnt)), average_duration) + randint(average_duration, size=sample_cnt)
            # Sample with random initial offset but keep the interval same
            selected_indices = np.multiply(list(range(sample_cnt)), average_duration) + randint(average_duration)
        else:
            # Not enough to sample from. Just return all frames (full clip) and use zero padding later
            selected_indices = np.array(range(total_frames))
        
        return selected_indices

    def _get_val_indices(self, total_frames, sample_cnt):
        """
        Similar to _sample_indices but for validation and test and so, no randomness is applied.
        """
        average_duration = total_frames // sample_cnt
        if average_duration > 0:
            # Sample the middle frame from each segment
            selected_indices = np.multiply(list(range(sample_cnt)), average_duration) + average_duration//2
        else:
            # Not enough to sample from. Just return all frames (full clip) and use zero padding later
            selected_indices = np.array(range(total_frames))
        
        return selected_indices

    def __getitem__(self, index):
        SAMPLE_CNT_VID = self.sample_cnt_vid
        SAMPLE_CNT_POSE = self.sample_cnt_pose
        
        # Pose tensor
        path_to_pose_pickle = self.data_path + self.split_name + '/' + self.sample_name[index]
        with open(path_to_pose_pickle, 'rb') as f:
            pose_data = pickle.load(f)
        total_pose_frames = pose_data.shape[1]
        
        if self.sampling_strategy == 'default': # Randomly sample, zero pad or trim.
            if self.split_name=='train':
                selected_indices_pose = self._sample_indices(total_pose_frames, SAMPLE_CNT_POSE)
            elif self.split_name=='validation' or self.split_name=='test_with_labels':
                selected_indices_pose = self._get_val_indices(total_pose_frames, SAMPLE_CNT_POSE)

            data_numpy = np.zeros((pose_data.shape[0],SAMPLE_CNT_POSE,pose_data.shape[2],pose_data.shape[3]), dtype=np.float32)
            for i in range(len(selected_indices_pose)):
                """For shorter clips, len(selected_indices_pose) < SAMPLE_CNT_POSE
                By default, the rest of the frames are zero-padded in that case.
                For longer clips, len(selected_indices_pose) > SAMPLE_CNT_POSE
                In that case, only the first SAMPLE_CNT_POSE frames are considered."""
                data_numpy[:, i, :, :] = pose_data[:, selected_indices_pose[i], :, :]

        elif self.sampling_strategy == 'crop_and_resize':
            # Sample a subsegment, and then temporally resize with interpolation
            selected_indices_pose = tools._crop_indices_subsegment(total_pose_frames, self.p_interval, self.pose_fps) # might return more than SAMPLE_CNT_POSE frames.
            
            # This is to get a fixed temporal sized pose data (e.g., 120 frames)
            data_numpy = np.zeros((pose_data.shape[0],len(selected_indices_pose),pose_data.shape[2],pose_data.shape[3]), dtype=np.float32)
            for i in range(len(selected_indices_pose)):
                data_numpy[:, i, :, :] = pose_data[:, selected_indices_pose[i], :, :]
            data_numpy = tools._resize_data_temporal(data_numpy, window_size=SAMPLE_CNT_POSE)
            
        pose_data_tensor = torch.from_numpy(data_numpy)

        
        D_feature = 1536 if self.rgb_feature_source=='dino' else 2048        
        rgb_data_numpy = np.zeros((SAMPLE_CNT_VID, D_feature), dtype=np.float32)

        if self.rgb_feature_source!='none': # Do not bother looking for RGB data in pose-only experiments
            # Pose data retrieved. Now move on to video tensor.
            path_to_jpg_list = self.rgb_data_path + self.split_name + '/' + self.sample_name[index]
            with open(path_to_jpg_list, 'rb') as f:
                jpg_list = pickle.load(f)
            
            ### Pick RGB frames based on the selected pose indices ###
            min_idx = selected_indices_pose[0]//2 # 60fps to 30 fps
            max_idx = selected_indices_pose[-1]//2
            interval = (max_idx-min_idx) / self.sample_cnt_vid # Preserve as float for better alignment
    
            if self.rgb_sampling_within_window=='first':
                selected_indices_vid = np.array([min_idx+int(i*interval) for i in range(self.sample_cnt_vid)])
            elif self.rgb_sampling_within_window=='mid':
                offset = min_idx + int(interval/2)
                selected_indices_vid = np.array([offset+int(i*interval) for i in range(self.sample_cnt_vid)])
            elif self.rgb_sampling_within_window=='random':
                init_T = int(interval)
                offset = (min_idx + np.random.randint(init_T)) if init_T>0 else min_idx
                selected_indices_vid = np.array([offset+int(i*interval) for i in range(self.sample_cnt_vid)])

            for i in range(len(selected_indices_vid)):
                lmdb_key = jpg_list[selected_indices_vid[i]]
                 
                if self.crop_scale in ['full', 'both']:
                    with self.env_fullImg.begin() as e:
                        # Full frame TSM or DINO feature
                        # Change camera name for TSM features if needed

                        # if self.rgb_feature_source=='tsm': # use it if any other view than v4 is used
                        #     lmdb_key = lmdb_key.replace('C10119_rgb', 'C10115_rgb')
                        if self.rgb_feature_source=='tsm_ego_e4': # if ego view tsm is needed, change the key first
                            lmdb_key = lmdb_key.replace('C10119_rgb', 'HMC_84358933_mono10bit')
                        elif self.rgb_feature_source=='tsm_ego_e3':
                            lmdb_key = lmdb_key.replace('C10119_rgb', 'HMC_84355350_mono10bit')
                        
                        data = e.get(lmdb_key.strip().encode('utf-8'))
                        if data is None:
                            # either actually not found, otherwise camera name is different for the egocentric ones
                            if self.rgb_feature_source=='tsm_ego_e4':
                                lmdb_key = lmdb_key.replace('HMC_84358933_mono10bit', 'HMC_21179183_mono10bit')
                                with self.env_fullImg_2.begin() as e2:
                                    data = e2.get(lmdb_key.strip().encode('utf-8'))
                            elif self.rgb_feature_source=='tsm_ego_e3':
                                lmdb_key = lmdb_key.replace('HMC_84355350_mono10bit', 'HMC_21110305_mono10bit')
                                with self.env_fullImg_2.begin() as e2:
                                    data = e2.get(lmdb_key.strip().encode('utf-8'))

                            # If still not found, write a error log file txt
                            if data is None:
                                # write a error log file txt
                                with open('error_log.txt', 'a') as f:
                                    f.write(f'[ERROR] Key {lmdb_key} does not exist !!!\n')
                            else:
                                data = np.frombuffer(data, 'float32') # convert to numpy array
                                rgb_data_numpy[i, :] = data 
                                                            
                        else:
                            data = np.frombuffer(data, 'float32')  # convert to numpy array
                            rgb_data_numpy[i, :] = data

                if self.crop_scale in ['cropped', 'both']:
                    # Cropped image DINO feature
                    with self.env_croppedImg.begin() as e:
                        data = e.get(lmdb_key.strip().encode('utf-8'))
                        if data is not None:
                            data = np.frombuffer(data, 'float32')  # convert to numpy array
                            rgb_data_numpy[i, :] += data ### Initial values are either zero or the full-sized image feature values.
            
        rgb_data_tensor = torch.from_numpy(rgb_data_numpy)

        label = self.label[index]
        verb_label = self.verb_label[index]
        noun_label = self.noun_label[index]
        

        # If the target is verb (noun), return verb (noun) label in place of the action label for easy evaluation.    
        if self.target_type=='verb':
            return pose_data_tensor, rgb_data_tensor, verb_label, verb_label, noun_label, index
        elif self.target_type=='noun':
            return pose_data_tensor, rgb_data_tensor, noun_label, verb_label, noun_label, index
        else:
            return pose_data_tensor, rgb_data_tensor, label, verb_label, noun_label, index    


    def top_k(self, score, top_k):
        rank = score.argsort()
        if self.target_type=='verb':
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.verb_label)]
        elif self.target_type=='noun':
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.noun_label)]
        else:
            hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

if __name__ == '__main__':
    # create an instance of Feeder class
    feeder = Feeder(data_path='/mnt/data/salman/Assembly101_AR_fusion_data/data/HAND_GCN/RAW_contex30_thresh0/',
                    rgb_data_path='/mnt/data/salman/Assembly101_AR_fusion_data/data/HAND_VID/RAW_contex30_thresh0/',
                    label_path='/mnt/data/salman/Assembly101_AR_fusion_data/data/Label_contex30_thresh0/',
                    split_name='train', target_type='action',
                    sample_cnt_vid=8, sample_cnt_pose=120,
                    sampling_strategy='crop_and_resize',
                    p_interval=[0.5,1.0],
                    rgb_feature_source='dino', crop_scale='both',
                    rgb_sampling_within_window='first')

    # get an individual item
    item = feeder[0]

    # print the item's data and label   
    print(item[0].shape) # data_tensor shape
    print(item[1].shape) # rgb_data_tensor shape
    print(item[2]) # label
    print(item[3]) # verb label
    print(item[4]) # noun label
    print(item[5]) # index
    print(item[6]) # total_pose_frames
