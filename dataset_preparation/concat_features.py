import os
import pandas as pd
import pdb
import torch
import numpy as np

feature1_list_file_path = '/home/mai/jchoi/src/TA3N/dataset_i3d/hmdb51/list_hmdb51_val_hmdb_ucf-feature.txt'
feature2_list_file_path = '/home/mai/jchoi/src/TA3N/dataset/hmdb51/list_hmdb51_val_hmdb_ucf-feature.txt'
concat_feature_output_path = '/net/acadia9a/data/jchoi/data/ucf_hmdb_full/TA3N/hmdb51/RGB-feature_i3d_resnet101'

# 1) make a dictionary: key is vid, values are i3d feature path, and resnet feature path
df = pd.read_csv(feature1_list_file_path, header=None, sep=' ')
data1 = df.to_numpy()

concat_dict = {}
for row in data1:
	vid = row[0].split('/')[-1]
	# if 'Goal_1_&_2_kick_ball_f_cm_np1_fr_goo_2' in vid:
	# 	pdb.set_trace()
	vid = vid.replace('&','')
	vid = vid.replace('(','')
	vid = vid.replace(')','')
	concat_dict[vid] = [row[0], row[1]] 

df = pd.read_csv(feature2_list_file_path, header=None, sep=' ')
data2 = df.to_numpy()

frames_diff_cnt = 0
for row in data2:
	vid = row[0].split('/')[-1]
	# if vid == 'Rapier_and_Rotella_Shield_Second_Bout_Nick_vs__Gareth_fencing_f_cm_np2_le_bad_4':
	# 	pdb.set_trace()
	concat_dict[vid].append(row[0])
	concat_dict[vid].append(row[1])
	if concat_dict[vid][1] != concat_dict[vid][3]:
		frames_diff_cnt += 1

if not os.path.exists(concat_feature_output_path):
	os.makedirs(concat_feature_output_path)

# pdb.set_trace()
# assert frames_diff_cnt == 0

# 2) read features for every key and concat features
for i,k in enumerate(concat_dict.keys()):
	if i%10 == 0:
		print('processing {}/{} videos'.format(i, len(concat_dict.keys())))
	# assert concat_dict[k][1] == concat_dict[k][3]
	max_num_frms = min(concat_dict[k][1], concat_dict[k][3])
	for frame in np.arange(1,max_num_frms+1):		
		feat1 = torch.load(concat_dict[k][0]+'/img_{:05d}.t7'.format(frame))
		feat2 = torch.load(concat_dict[k][2]+'/img_{:05d}.t7'.format(frame))
		feat_final = torch.cat([feat2, feat1])
		
		cur_cls = concat_dict[k][0].split('/')[-2]
		cur_vid = concat_dict[k][0].split('/')[-1]
		feat_final_path = os.path.join(concat_feature_output_path, cur_cls, cur_vid)
		
		if not os.path.exists(feat_final_path):
			os.makedirs(feat_final_path)

		torch.save(feat_final.clone(), os.path.join(feat_final_path, 'img_{:05d}.t7'.format(frame)))	

print('Concatenation done')