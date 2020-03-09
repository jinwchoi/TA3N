import os
import pandas as pd
import pdb

feature1_list_file_path = '/home/mai/jchoi/src/TA3N/dataset_i3d/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'
feature2_list_file_path = '/home/mai/jchoi/src/TA3N/dataset/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'

# 1) make a dictionary: key is vid, values are i3d feature path, and resnet feature path
df = pd.read_csv(feature1_list_file_path, header=None, sep=' ')
data1 = df.to_numpy()

concat_dict = {}
for row in data1:
	vid = row[0].split('/')[-1]
	concat_dict[vid] = [row[0], row[1]] 

df = pd.read_csv(feature2_list_file_path, header=None, sep=' ')
data2 = df.to_numpy()

frames_diff_cnt = 0
for row in data2:
	vid = row[0].split('/')[-1]
	concat_dict[vid].append(row[0])
	concat_dict[vid].append(row[1])
	if concat_dict[vid][1] != concat_dict[vid][3]:
		frames_diff_cnt += 1

pdb.set_trace()
# 2) read features for every key

# 3) concatenate feature

# 4) save the feature

print('')