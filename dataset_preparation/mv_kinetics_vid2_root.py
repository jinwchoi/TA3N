import os
import pdb

rootdir = '/net/acadia9a/data/jchoi/data/kinetics/RGB-feature3_i3d'
# rootdir = '/net/acadia9a/data/jchoi/data/kinetics/test_rgb'

listsub_dirs = ['train', 'val']

# train or val
for subdir in listsub_dirs:
	list_cls_dirs = os.listdir(os.path.join(rootdir,subdir))
	for cur_cls_dir in list_cls_dirs:
		cur_path = os.path.join(rootdir,subdir,cur_cls_dir)
		print(cur_path)
		cmd = 'mv "{}"/* "{}"/../../'.format(cur_path,cur_path)
		print(cmd)
		os.system(cmd)