import os
import pandas as pd
import pdb
import torch
import numpy as np

feature1_list_file_path = '/home/mai/jchoi/src/TA3N/dataset_i3d/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'
feature2_list_file_path = '/home/mai/jchoi/src/TA3N/dataset/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'
concat_feature_output_path = '/net/acadia9a/data/jchoi/data/ucf_hmdb_full/TA3N/ucf101/RGB-feature_i3d_resnet101'

concat_list_file_path = '/home/mai/jchoi/src/TA3N/dataset_i3d/ucf101/list_ucf101_val_hmdb_ucf-feature-concat-i3d-resnet101.txt'

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

	concat_dict[vid] = [row[0], row[1], row[2]] 

df = pd.read_csv(feature2_list_file_path, header=None, sep=' ')
data2 = df.to_numpy()

frames_diff_cnt = 0
for row in data2:
	vid = row[0].split('/')[-1]
	# if vid == 'Rapier_and_Rotella_Shield_Second_Bout_Nick_vs__Gareth_fencing_f_cm_np2_le_bad_4':
	# 	pdb.set_trace()
	concat_dict[vid].append(row[0])
	concat_dict[vid].append(row[1])
	if concat_dict[vid][1] != concat_dict[vid][4]:
		frames_diff_cnt += 1

if not os.path.exists(concat_feature_output_path):
	os.makedirs(concat_feature_output_path)

# 2) read features for every key and concat features
rows = []
for i,k in enumerate(concat_dict.keys()):
	cur_cls = concat_dict[k][0].split('/')[-2]
	cur_vid = concat_dict[k][0].split('/')[-1]
	feat_final_path = os.path.join(concat_feature_output_path, cur_cls, cur_vid)
	
	num_frms = min(concat_dict[k][1],concat_dict[k][4])
	
	# cur_row = feat_final_path + ' {}'.format(num_frms) + ' {}'.format(concat_dict[k][2])
	cur_row = [feat_final_path, num_frms, concat_dict[k][2]]
	rows.append(cur_row)

df = pd.DataFrame(np.array(rows))
df.to_csv(concat_list_file_path, header=None, index=None, sep=' ')

print('Conversion done')