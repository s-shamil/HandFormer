"""
This script will generate a pickle file for each segment or action clip.
Segment: fine-grained actions in a video, denoted by start_frame and end_frame.

NOTE: Annotations are 30fps. If 60 fps pose/video is used, double the frame id.
"""


### Libraries
import numpy as np
import pandas as pd
import json
import os
import pickle
import statistics
import sys
import torch
from tqdm import tqdm
from operator import itemgetter

### Configuration params and custom util functions
import config
from utils import *


def gather_split_annotations(annotations_split, all_frames_dict, jsons_list_poses):
    """
    Given an annotation_split (train/val/test), this function populates <all_frames_dict>.
    <all_frames_dict> is a dictionary keeping a list of segments against every video id, i.e., all action clips in a particular video together in a list.
    Parameters:
    - annotations_split: DataFrame, annotation split (train/val/test)
    - all_frames_dict: Dictionary, to store all action clips against individual video id
    - jsons_list_poses: List, list of json files with hand poses. Ideally, there is one json file for the whole video.
    """

    # Counting total number of segments in all_frames_dict
    total_segments_csv = 0 # Total number of segments in the annotation csv file
    vi_segments_csv = 0 # Number of segments in the csv file for the specific view v_i
    vi_segments_with_pose = 0 # Number of segments in the csv file for the specific view v_i with pose available
    
    for _, aData in tqdm(annotations_split.iterrows(), 'Populating Dataset', total=len(annotations_split)):
        total_segments_csv += 1

        # Taking view 4 (v4) annotations in Assembly101. Just to ensure that we only read an action clip once, the particular view doesn't matter.
        if (aData.video).endswith("C10119_rgb.mp4"):
            vi_segments_csv += 1

            # For each segment, find the video id first
            video_id_json = aData.video.split('/')[0] + '.json'

            # If the hand poses for the video is not available, skip it
            if not video_id_json in jsons_list_poses:
                continue
            
            # Pose data available
            vi_segments_with_pose += 1
            # Store segment information as a dictionary
            curr_data = dict()
            curr_data['start_frame'] = aData.start_frame
            curr_data['end_frame'] = aData.end_frame
            curr_data['action'] = aData.action_id
            curr_data['noun'] = aData.noun_id
            curr_data['verb'] = aData.verb_id
            curr_data['action_cls'] = aData.action_cls
            curr_data['toy_id'] = aData.toy_id

            # Add the dictionary to the list of action clips for the video
            all_frames_dict[video_id_json].append(curr_data)

    print("Total number of entries in csv: ", total_segments_csv)
    print("Number of unique segments: ", vi_segments_csv)
    print("Number of uniqye segments with pose: ", vi_segments_with_pose)


def total_hand_frames(jsons_list_poses, files_dir):
    """
    To check the total number of frames available in hand pose json files
    """
    total_sum = 0
    for annotation_file in tqdm(jsons_list_poses, total=len(jsons_list_poses)):
        tqdm.write(annotation_file)
        with open(files_dir + annotation_file) as BOB:
            hand_labels = json.load(BOB)
        total_sum += len(hand_labels)
    print(total_sum)


### Full video data readers. Will read pose for the whole video together instead of clip by clip.
def getPose3d(pose3d_json, camera_pose_json):
    """
    Read pose3d_json and camera_pose_json
    Return two numpy arrays:
    pose3d_full: (L, 2, 21, 3) shaped array. L: #frames in video. 2: Left and Right hands. 21: 21 joints. 3: x,y,z coordinates
    skel3d: two hands along with head position denoted from camera pose. (L, 1, 43, 3) shaped array
    """
    with open(pose3d_json, 'r') as f:
        hand_data = json.load(f)

    with open(camera_pose_json, 'r') as f2:
        cam_data = json.load(f2)
    
    frame_id_intArr = [int(x) for x in hand_data.keys()]
    frame_cnt_total = max(len(hand_data.keys()), np.max(frame_id_intArr)+1)
    pose3d_full = np.zeros((frame_cnt_total, 2, 21, 3))
    skel3d = np.zeros((frame_cnt_total, 1, 43, 3))

    for frame_id in hand_data.keys():
        frame_data = hand_data[frame_id]
        frame_data_L = frame_data['0']
        frame_data_R = frame_data['1']
        frame_data_L = np.array(frame_data_L)
        frame_data_R = np.array(frame_data_R)
        # Shape of frame_data_L and frame_data_R: (21,3)

        # Find head position from camera poses
        try:
            cam_position_data = cam_data[frame_id]
            head_pos = [0.0, 0.0, 0.0] # Put the average of the 4 cameras
            cam_cnt = 0
            for cam_key in cam_position_data.keys():
                loc = cam_position_data[cam_key]
                head_pos = np.add(head_pos, loc)
                cam_cnt += 1
            head_pos /= cam_cnt

        except:
            # Camera pose not available for the frame. Put the average of the wrist positions. 
            head_pos = (frame_data_L[5] + frame_data_R[5])/2 # 5th joint is wrist

        fid = int(frame_id)
        pose3d_full[fid, 0, :, :] = frame_data_L
        pose3d_full[fid, 1, :, :] = frame_data_R
        skel3d[fid, :, 0:21, :] = frame_data_L
        skel3d[fid, :, 21:42, :] = frame_data_R
        skel3d[fid, :, 42:43, :] = head_pos
        
    return pose3d_full, skel3d

def getXfTransform(xf_transform_json):
    # XF transformation matrices will convert the joint locations. Target coordinate system will follow either of the wrist joints.
    # Read xf_transform_json to make L, 2, 4, 4 shaped numpy array
    with open(xf_transform_json, 'r') as f:
        data = json.load(f)
        frame_id_intArr = [int(x) for x in data.keys()]
        frame_cnt_total = max(len(data.keys()), np.max(frame_id_intArr)+1)
        xf_trans_full = np.zeros((frame_cnt_total, 2, 4, 4))
        for frame_id in data.keys():
            frame_data = data[frame_id]
            frame_data_L = frame_data['0']
            frame_data_R = frame_data['1']
            frame_data_L = np.array(frame_data_L)
            frame_data_R = np.array(frame_data_R)
            # Shape of frame_data_L and frame_data_R: (4,4)

            fid = int(frame_id)
            xf_trans_full[fid, 0, :, :] = frame_data_L
            xf_trans_full[fid, 1, :, :] = frame_data_R

    return xf_trans_full


def main():

    contex_val = config.num_frames_in_each_context
    valid_hand_thresh = config.handpose_min_confidence
    # Directory suffix for saving raw data (one pickle for each action clip)
    dir_suffix = 'RAW_contex' + str(contex_val) + '_thresh' + str(int(valid_hand_thresh)) + '/'

    pose3d_dir = config.raw_pose_dir # Directory of raw hand poses (json files for each video)
    camera_pos_dir = config.camera_pose_dir # Directory of camera poses (json files for each video). Used to find head position.
    xf_transform_dir = config.xf_transform_dir # Directory of xf transformation matrices for hand poses
    rgb_root_path = config.rgb_frames_root_dir # Directory of rgb frames. Can be empty for pose only projects.
    POSE_ONLY_FLAG = (rgb_root_path == '') # If True, only pose data will be saved ignoring rgb frames.
    CAMERA_NAME = config.camera_name # Which camera to use for rgb input

    path_to_csv = config.annot_dir # Contains csv files for train/test/validation splits (each line indicating a segment and its annotation)
    save_pos_path = config.out_path_for_handpose + dir_suffix # Output path for storing instance wise pose data, no head position
    save_pos_path_OneSkeleton = config.out_path_for_handpose_with_head + dir_suffix # Output path for storing instance wise pose data, with head position. 21*2+1 joints
    save_vid_path = config.out_path_for_rgb + dir_suffix # Output path for storing instance wise video data.
                                                            # Can simply be a list of rgb frames to feed later.
    
    
    print('Saving pose output into ---> ' + save_pos_path)
    print('==========================================')
    print('Saving pose + head position (one skeleton) output into ---> ' + save_pos_path_OneSkeleton)
    print('==========================================')
    print('Saving video output into ---> ' + save_vid_path)
    print('==========================================')
 
    # Take hand pose files and sort on names
    jsons_list_poses = os.listdir(pose3d_dir)
    jsons_list_poses.sort()

    flag_print = False
    if flag_print:
        total_hand_frames(jsons_list_poses, pose3d_dir)

    # Column names of the csv files
    name_list = ['id', 'video', 'start_frame', 'end_frame', 'action_id', 'verb_id', 'noun_id',
                    'action_cls', 'verb_cls', 'noun_cls', 'toy_id', 'toy_name', 'is_shared', 'is_RGB']

    
    list_val_type = ['train', 'validation', 'test_with_labels']
    
    max_frame = [] # Keep records for number of handpose frames per segment
    
    for kll in range(len(list_val_type)):
        split_type = list_val_type[kll] # 'test', 'validation', 'train'
        counter_instances = 0 # Instance number: Increment after appending data for each segment

        pose_save_out = save_pos_path + split_type
        skeleton_save_out = save_pos_path_OneSkeleton + split_type
        vid_save_out = save_vid_path + split_type
        if not os.path.exists(pose_save_out):
            os.makedirs(pose_save_out)
        if not os.path.exists(skeleton_save_out):
            os.makedirs(skeleton_save_out)
        if not os.path.exists(vid_save_out):
            os.makedirs(vid_save_out)

        annotations_ = pd.read_csv(path_to_csv + split_type + ".csv", header=0, low_memory=False, names=name_list)
        all_frames_dict = dict() # all_frames_dict is a dictionary keeping a list of clips against every video id
        
        for kli in range(len(jsons_list_poses)):
            # Initiate empty list for each video id in  all_frames_dict
            all_frames_dict[jsons_list_poses[kli]] = []
        
        # Populate all_frames_dict
        gather_split_annotations(annotations_, all_frames_dict, jsons_list_poses)

        # Generate output data
        for klk in range(len(jsons_list_poses)):
            # Get the list of action clips for one video
            all_segments = all_frames_dict[jsons_list_poses[klk]]
            if len(all_segments) == 0: # no action clips in this video for the current split
                continue
            # Sort the segments based on start_frame
            all_segments = sorted(all_segments, key=itemgetter('start_frame'))

            # Read the corresponding video file's name
            vid_file_name = jsons_list_poses[klk][:-5] # Discarding last 5 characters (.json)
            
            print("klk index = {} | Video Name: {}.".format(klk, vid_file_name))
            
            # Read pose file with xf transform and video frames
            pose3d_json = pose3d_dir + vid_file_name + '.json'
            xf_transform_json = xf_transform_dir + vid_file_name + '.json'
            camera_pos_json = camera_pos_dir + vid_file_name + '.json'

            if not POSE_ONLY_FLAG:
                vid_file_frame_dir = rgb_root_path + vid_file_name + '/'+ CAMERA_NAME + '/'

            pose3d_full, skel3d_full = getPose3d(pose3d_json, camera_pos_json)
            xf_trans_full = getXfTransform(xf_transform_json)

            if not POSE_ONLY_FLAG:
                available_rgb_frames = len(os.listdir(vid_file_frame_dir)) # Can discard it if not needed, e.g., for pose only projects
            else:
                available_rgb_frames = 0

            print(klk, '/', len(jsons_list_poses), '  - #seg:', len(all_segments), '  - #pose:', len(pose3d_full), '  - #frames:', available_rgb_frames)

            for ikl, segment in enumerate(tqdm(all_segments)):
                # For each segment, we will store handposes, skeletons and rgb frames into separate pickle files
                hand_joints_seg = [] # Stores all handposes (2,21) for current segment
                skeletons_seg = [] # Stores one skeleton versions with head (1,21*2+1) of the current segment
                xf_trans_seg = [] # Stores all transformation metrices for current segment
                
                rgb_frame_paths = [] # Store frame paths in this array. Later Feeder will retrieve rgb or tsm or dino features.

                # 30 fps annotation is used.
                # Safe indexing to avoid error. --> Start 1-indexed. End 0-indexed
                start_f = max(1, segment['start_frame'] - contex_val) # Adjust start_frame with context
                if not POSE_ONLY_FLAG:
                    end_f = min(segment['end_frame'] + contex_val + 1, available_rgb_frames) # Adjust end_frame with context
                else:
                    end_f = segment['end_frame'] + contex_val + 1 # No RGB information, so keep looking for pose frames.
                for img_index in range(start_f, end_f, 1):
                    # Dealing with 60 fps pose data with 30fps annotations
                    pose_ended = False

                    for offset in range(2): # 0,1
                        take_index = img_index*2 + offset
                        
                        if take_index >= len(pose3d_full):
                            # no more pose data available in this clip
                            pose_ended = True
                            break

                        landmarks3d = np.zeros((2, 21, 3), dtype='float32') # 3D coordinates for 21 landmarks for each of the 2 hands
                        xf_trans_twoHands = np.zeros((2,4,4), dtype='float32')
                        skeleton3d = skel3d_full[take_index].astype(np.float32)

                        for hand in range(0, 2):
                            curr_hand_pose = pose3d_full[take_index][hand]
                            hand_landmarks3d = np.array(curr_hand_pose, dtype='float32') # Shape: (21,3)
                            transform_4x4 = np.array(xf_trans_full[take_index][hand], dtype='float32') # Shape: (4,4)
                            
                            if valid_hand_thresh > 0:
                                print("Confidence thresholding not implemented. Coming soon...!")
                                sys.exit()
                            else:
                                landmarks3d[hand] = hand_landmarks3d
                                xf_trans_twoHands[hand] = transform_4x4
                                
                        hand_joints_seg.append(landmarks3d)
                        skeletons_seg.append(skeleton3d)
                        xf_trans_seg.append(xf_trans_twoHands)

                    if not POSE_ONLY_FLAG:
                        # Retrieve the rgb frame
                        frame_path = vid_file_frame_dir + CAMERA_NAME + "_{:010d}".format(img_index) + '.jpg'
                        rgb_frame_paths.append(frame_path)
                    
                    if pose_ended:
                        break

                rgb_len = len(rgb_frame_paths)
                
                if (len(hand_joints_seg) > 0):
                    hand_joints_seg_tensor = torch.from_numpy(np.array(hand_joints_seg)) # L, 2, 21, 3
                    skeletons_seg_tensor = torch.from_numpy(np.array(skeletons_seg)) # L, 1, 43, 3
                    xf_trans_seg_tensor = torch.from_numpy(np.array(xf_trans_seg)) # L, 2, 4, 4

                    hand_joints_seg_tensor = hand_joints_seg_tensor.unsqueeze(0)
                    skeletons_seg_tensor = skeletons_seg_tensor.unsqueeze(0)
                    xf_trans_seg_tensor = xf_trans_seg_tensor.unsqueeze(0)

                    hand_joints_seg_transformed = get_xf_transformed_data(hand_joints_seg_tensor, xf_trans_seg_tensor).numpy()
                    skeletons_seg_transformed = get_xf_transformed_data(skeletons_seg_tensor, xf_trans_seg_tensor).numpy()
                    hand_joints_seg_s = hand_joints_seg_transformed[0]
                    skeletons_seg_s = skeletons_seg_transformed[0]


                    # Segment name with instance number, action_id, verb_id, noun_id and length of the segment
                    seg_name = split_type[:2] + str(counter_instances) + '_' + 'a' + str(segment['action']) + \
                            '_v' + str(segment['verb']) + '_n' + str(segment['noun']) + \
                            '_lenPose' + str(hand_joints_seg_s.shape[1]) + '_lenRGB' + str(rgb_len)

                    # Output file name with segment name and action name
                    save_pose_file = pose_save_out + '/' + seg_name + '_' + segment['action_cls'].replace(' ', '_') + '.pkl'
                    save_skeleton_file = skeleton_save_out + '/' + seg_name + '_' + segment['action_cls'].replace(' ', '_') + '.pkl'
                    if not POSE_ONLY_FLAG:
                        save_vid_file = vid_save_out + '/' + seg_name + '_' + segment['action_cls'].replace(' ', '_') + '.pkl'
                    
                    # Dump the (3, T, 21, 2) sized numpy array for handposes for each segment
                    with open(save_pose_file, 'wb') as handle:
                        pickle.dump(hand_joints_seg_s, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # Dump the (3, T, 43, 1) sized numpy array for handposes for each segment
                    with open(save_skeleton_file, 'wb') as handle:
                        pickle.dump(skeletons_seg_s, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if not POSE_ONLY_FLAG:
                        # Dump the (T_,) sized list of rgb paths
                        with open(save_vid_file, 'wb') as handle:
                            pickle.dump(rgb_frame_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    max_frame.append(hand_joints_seg_s.shape[1]) # Segment wise frame count
                    counter_instances += 1

        max_frame.sort()
        print(split_type, '#segments', counter_instances, ' --- durations (in frames) max=', max(max_frame),
              ', min=', min(max_frame), ', med=', statistics.median(max_frame))


if __name__ == '__main__':
    main()
