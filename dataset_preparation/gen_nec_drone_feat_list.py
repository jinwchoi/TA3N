# generate the file list from video dataset (TODO: update the code w/ DA_setting)
import os
import glob as gb
import numpy as np
import pandas as pd
import pdb

phase = 'val' # 'train' or 'val' or 'test'
necdrone_feature_dir = '/net/acadia9a/data/jchoi/data/nec_drone/2018/TA3N/RGB-feature_resnet101'
necdrone_anno_path = '/net/acadia7a/data/jchoi/NEC-Drone/Annotation/NEC-Drone-7_Annotation_10102018_{}list.csv'.format(phase)
classfile_path = '/net/acadia9a/data/jchoi/data/kinetics/anno/K7-ND7-classes.txt'
output_anno_path = '/home/mai/jchoi/src/TA3N/dataset/necdrone/list_necdrone_{}_kinetics_necdrone-feature.txt'.format(phase)

df = pd.read_csv(necdrone_anno_path, header=None)
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
	cur_id = '_'.join(cur_path[-4:])
	cur_label = int(anno[-2])
	cur_num_frames_anno = int(anno[2]) - int(anno[1]) + 1
	
	cur_feat_path = os.path.join(necdrone_feature_dir, cur_id)
	featfile_list = os.listdir(cur_feat_path)

	if cur_num_frames_anno != len(featfile_list):
		not_identical_vid_ids.append(cur_id)

	cur_output_anno = 'dataset/necdrone/RGB-feature_resnet/' + cur_id  + ' {} {}'.format(len(featfile_list), cur_label)
	output_anno.append(cur_output_anno)

output_anno = np.array(output_anno)
df = pd.DataFrame(output_anno)
df.to_csv(output_anno_path, header=None, index=None)

print('')
