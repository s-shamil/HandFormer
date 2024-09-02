num_frames_in_each_context = 30 # e.g., 15 frames (at 30fps) will add 0.5+0.5=1 seconds of context in total. Use 30 for 2 seconds of context in total. 
handpose_min_confidence = 0.0 # Minimum confidence value to include a handpose frame in the list

raw_pose_dir = "/mnt/data/salman/assembly101_poses/assembly101_camera_and_hand_poses/landmarks3D/"
camera_pose_dir = "/mnt/data/salman/assembly101_poses/assembly101_camera_and_hand_poses/camera_position_ego/"
xf_transform_dir = "/mnt/data/salman/assembly101_poses/assembly101_camera_and_hand_poses/xf_transf/"
camera_name = 'C10119_rgb'
rgb_frames_root_dir = "" # keep empty if rgb frames are not available on machine. 

annot_dir = '/mnt/data/salman/Assembly101_FG_annotations/'
out_path_for_handpose = '/mnt/data/salman/Assembly101_AR_fusion_data/data/HAND_GCN/' 
out_path_for_handpose_with_head = '/mnt/data/salman/Assembly101_AR_fusion_data/data/HAND_GCN_OneSkeleton/'
out_path_for_rgb = '/mnt/data/salman/Assembly101_AR_fusion_data/data/HAND_VID/'

final_data_path = '/mnt/data/salman/Assembly101_AR_fusion_data/data/'