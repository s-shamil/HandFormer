"""
This script will generate data (X) and labels (Y) for each split (train/val/test) from the segment wise pickle files.
"""
import os
import pickle

import numpy as np
from tqdm import tqdm

import config



def gendata(data_path, skel_data_path, rgb_data_path, out_label_path, part='train'):
    """
    This function will generate data (X) and labels (Y) for each split (train/val/test).
    X will contain the list of the pickle files.
    Y will contain the action labels for each segment. 
    """

    if not os.path.exists(out_label_path):
        os.mkdir(out_label_path)

    sample_name = [] # Stores segment identifier, which is basically the filenames of the pickles 
                     # each pickle has (3, T, 21, 2) sized numpy array for certain segment
                     # T = number of frames in the segment. Feeders will have to handle variable length sequences.

    sample_label = [] # Action label for segment
    num_frames = [] # Number of pose frames in segments

    for filename in tqdm(os.listdir(data_path)):
        action_class = int(filename.split('_')[1].split('a')[1])  # action
        # verb_class = int(filename.split('_')[2].split('v')[1]) # verb
        # noun_class = int(filename.split('_')[3].split('n')[1]) # verb

        sample_name.append(filename) 
        sample_label.append(action_class) 
        # sample_label.append(verb_class)
        # sample_label.append(noun_class)
        num_frames.append(int(filename.split('_')[4].split('lenPose')[1]))

    # Stats on number of frames for the segments
    list_frames = np.array(num_frames)

    print(len(set(sample_label)), "<- Number of distinct actions present in this split")

    print('max=', np.max(list_frames), ', min=', np.min(list_frames),
          ', mean=', int(np.mean(list_frames)), ', med=', np.median(list_frames))

    # Save action labels against file names in a pickle (Y part).
    with open('{}/{}_label.pkl'.format(out_label_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    


if __name__ == '__main__':

    main_path = config.out_path_for_handpose # hand poses generated by 1st script
    skeleton_path = config.out_path_for_handpose_with_head # hand poses with head forming a single skeleton
    # for rgb frames, the value will be None when no rgb frame is available on the machine
    rgb_main_path = config.out_path_for_rgb if len(config.rgb_frames_root_dir) > 0 else None 
    data_root = config.final_data_path # Output path for the final data
    
    contex_val = config.num_frames_in_each_context
    valid_hand_thresh = config.handpose_min_confidence
    data_dir_suffix = 'RAW_contex' + str(contex_val) + '_thresh' + str(int(valid_hand_thresh)) + '/'


    # Input
    data_path = main_path + data_dir_suffix
    skel_data_path = skeleton_path + data_dir_suffix
    rgb_data_path = (rgb_main_path + data_dir_suffix) if rgb_main_path is not None else None
    # Output
    out_label_path = data_root + 'Label' + data_dir_suffix[3:] # 3: -> Skip 'RAW'

    part = ['train', 'validation', 'test_with_labels']

    for p in part:
        data_path_ = os.path.join(data_path, p)
        skel_data_path_ = os.path.join(skel_data_path, p)
        rgb_data_path_ = os.path.join(rgb_data_path, p)
        # Generate data for the split
        gendata(data_path_, skel_data_path_, rgb_data_path_, out_label_path, part=p)