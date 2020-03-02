#!/bin/bash
# ----------------------------------------------------------------------------------
# variable
dataset=kinetics # # depend on users (e.g. hmdb51 | ucf101 | xxx_train | xxx_val) 
data_path=/net/acadia9a/data/jchoi/data/ # depend on users
video_in=vids
frame_in=RGB-feature_resnet101
max_num=-1 # 0 (class average) | -1 (all) | any number
random_each_video=N # Y | N

# method_read: affect the loaded frame numbers
# video: load from the raw video folder (slower, but more accurate)
# frame: load from the feature folder
method_read=frame # video | frame
frame_type=feature # frame | feature
DA_setting=kinetics_necdrone # hmdb_ucf | hmdb_phav | ps_kinetics | kinetics_phav | ucf_olympic
suffix='_'$DA_setting'-'$frame_type'_'$max_num

python video_dataset2list.py $dataset --data_path $data_path \
--video_in $video_in --frame_in $frame_in --max_num $max_num \
--class_select --DA_setting $DA_setting --suffix $suffix \
--random_each_video=$random_each_video --method_read $method_read

#----------------------------------------------------------------------------------
exit 0
