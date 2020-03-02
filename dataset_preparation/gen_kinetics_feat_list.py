# generate the file list from video dataset (TODO: update the code w/ DA_setting)
import os
import glob as gb
import numpy as np
import pandas as pd
import pdb

phase = 'val' # 'train' or 'val'
kinetics_feature_dir = '/net/acadia9a/data/jchoi/data/kinetics/RGB-feature_resnet101'
kinetics_anno_path = '/net/acadia9a/data/jchoi/data/kinetics/anno/kinetics-7-{}.csv'.format(phase)
classfile_path = '/net/acadia9a/data/jchoi/data/kinetics/anno/K7-ND7-classes.txt'
output_anno_path = '/home/mai/jchoi/src/TA3N/dataset/kinetics_splits/list_kinetics_val-feature.txt'

df = pd.read_csv(kinetics_anno_path, header=None)
data = df.to_numpy()

# df = pd.read_csv(classfile_path, header=None, sep=' ')
# class_data = df.to_numpy()
# label2cls 


not_identical_vid_ids = []
output_anno = []

for i,anno in enumerate(data):
	if i%100==0:
		print('Processing {}/{}'.format(i, len(data)))
		print(len(not_identical_vid_ids))
	cur_path = anno[0].split('/')
	cur_id = cur_path[-1].split('.')[0]
	cur_cls = cur_path[-2]
	cur_label = int(anno[-1])
	cur_num_frames_anno = int(anno[2]) - int(anno[1]) + 1
	
	cur_feat_path = os.path.join(kinetics_feature_dir, cur_id)
	featfile_list = os.listdir(cur_feat_path)

	if cur_num_frames_anno != len(featfile_list)-1:
		not_identical_vid_ids.append(cur_id)

	cur_output_anno = 'dataset/kinetics/RGB-feature_resnet/' + cur_id  + ' {} {}'.format(len(featfile_list), cur_label)
	# cur_output_anno = 'dataset/kinetics/RGB-feature_resnet/' + cur_id  + ' {} {}'.format(cur_num_frames_anno, cur_label)
	output_anno.append(cur_output_anno)
	
output_anno = np.array(output_anno)
df = pd.DataFrame(output_anno)
df.to_csv(output_anno_path, header=None, index=None)

pdb.set_trace()


print('')
