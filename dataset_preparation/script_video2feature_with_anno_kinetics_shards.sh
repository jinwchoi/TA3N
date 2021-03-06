#!/bin/bash -l
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --constraint="TITANX|TITANXP|GTX1080Ti"
##SBATCH --constraint="K80"
#SBATCH --cpus-per-task 5
#SBATCH --time 144:00:00
#SBATCH -J i3d-ktr0
#SBATCH -o /net/acadia9a/data/jchoi/data/nec_drone/2018/log/Kinetics-7-train-3d-feat_extraction_20200313_cls5_shard3of10.log

pwd; hostname; date
echo $CUDA_VISIBLE_DEVICES

source activate ta3n
which python

# ----------------------------------------------------------------------------------
# variable
data_path=/net/acadia9a/data/jchoi/data/kinetics/vids/ # depend on users: UCF: /net/acadia7a/data/public_datasets/UCF101/UCF-101/, HMDB: /net/acadia9a/data/jchoi/data/hmdb/videos/

video_in=train
feature_in=/net/acadia9a/data/jchoi/data/kinetics/RGB-feature3
input_type=video # video | frames
structure=tsn # tsn | imagenet
num_thread=4
batch_size=64 # need to be larger than 16 for c3d
base_model=i3d # resnet101 | c3d
pretrain_weight=/models/c3d.pickle # depend on users (only used for C3D model)
start_class=5 # start from 0 end at 12
end_class=5 # -1: process all the categories
cur_shard=3
total_shards=10
shard_start_idx=780
class_file=/net/acadia9a/data/jchoi/data/kinetics/anno/K7-ND7-classes.txt # none | XXX/class_list_DA.txt (depend on users)
anno_file=/net/acadia9a/data/jchoi/data/kinetics/anno/kinetics-7-train.csv 

python -W ignore video2feature_with_anno_v2.py --data_path $data_path --video_in $video_in \
--feature_in $feature_in --input_type $input_type --structure $structure \
--num_thread $num_thread --batch_size $batch_size --base_model $base_model --pretrain_weight $pretrain_weight \
--start_class $start_class --end_class $end_class --class_file $class_file --anno_file $anno_file --cur_shard $cur_shard --total_shards $total_shards --shard_start_idx $shard_start_idx


#----------------------------------------------------------------------------------
exit 0
