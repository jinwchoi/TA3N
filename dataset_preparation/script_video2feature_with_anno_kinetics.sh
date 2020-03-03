#!/bin/bash -l
#SBATCH --mem=128gb
#SBATCH --gres=gpu:8
##SBATCH --constraint="TITANX|TITANXP|GTX1080Ti"
#SBATCH --constraint="K80"
#SBATCH --cpus-per-task 17
#SBATCH --time 144:00:00
#SBATCH -J i3d
#SBATCH -o /net/acadia9a/data/jchoi/data/nec_drone/2018/log/Kinetics-7-train-i3d-feat_extraction_20200302.log

pwd; hostname; date
echo $CUDA_VISIBLE_DEVICES

source activate ta3n
which python

# ----------------------------------------------------------------------------------
# variable
data_path=/net/acadia9a/data/jchoi/data/kinetics/vids/ # depend on users: UCF: /net/acadia7a/data/public_datasets/UCF101/UCF-101/, HMDB: /net/acadia9a/data/jchoi/data/hmdb/videos/

video_in=train
feature_in=/net/acadia9a/data/jchoi/data/kinetics/RGB-feature
input_type=video # video | frames
structure=tsn # tsn | imagenet
num_thread=8
batch_size=64 # need to be larger than 16 for c3d
base_model=i3d # resnet101 | c3d
pretrain_weight=/models/c3d.pickle # depend on users (only used for C3D model)
start_class=1 # start from 1
end_class=-1 # -1: process all the categories
class_file=/net/acadia9a/data/jchoi/data/kinetics/anno/K7-ND7-classes.txt # none | XXX/class_list_DA.txt (depend on users)
anno_file=/net/acadia9a/data/jchoi/data/kinetics/anno/kinetics-7-train.csv

python -W ignore video2feature_with_anno.py --data_path $data_path --video_in $video_in \
--feature_in $feature_in --input_type $input_type --structure $structure \
--num_thread $num_thread --batch_size $batch_size --base_model $base_model --pretrain_weight $pretrain_weight \
--start_class $start_class --end_class $end_class --class_file $class_file --anno_file $anno_file

#----------------------------------------------------------------------------------
exit 0
