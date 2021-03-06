import argparse
from colorama import init
from colorama import Fore, Back, Style
import os
import imageio
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

import time

# for extracting feature vectors
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import pandas as pd
import pdb
import torch.nn.functional as F


imageio.plugins.ffmpeg.download()
init(autoreset=True)

###### Flags ######
parser = argparse.ArgumentParser(description='Dataset Preparation')
parser.add_argument('--data_path', type=str, required=False, default='', help='source path')
parser.add_argument('--video_in', type=str, required=False, default='RGB', help='name of input video dataset')
parser.add_argument('--feature_in', type=str, required=False, default='RGB-feature', help='name of output frame dataset')
parser.add_argument('--input_type', type=str, default='video', choices=['video', 'frames'], help='input types for videos')
parser.add_argument('--structure', type=str, default='tsn', choices=['tsn', 'imagenet'], help='data structure of output frames')
parser.add_argument('--base_model', type=str, required=False, default='resnet101', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'c3d', 'i3d'])
parser.add_argument('--pretrain_weight', type=str, required=False, default='', help='model weight file path')
parser.add_argument('--num_thread', type=int, required=False, default=-1, help='number of threads for multiprocessing')
parser.add_argument('--batch_size', type=int, required=False, default=1, help='batch size')
parser.add_argument('--start_class', type=int, required=False, default=0, help='the starting class id (start from 1)')
parser.add_argument('--end_class', type=int, required=False, default=-1, help='the end class id')
parser.add_argument('--cur_shard', type=int, required=False, default=0, help='current shard out of total shard')
parser.add_argument('--total_shards', type=int, required=False, default=1, help='total number of shards')
parser.add_argument('--shard_start_idx', type=int, required=False, default=1, help='start index')
parser.add_argument('--class_file', type=str, default='class.txt', help='process the classes only in the class_file')
parser.add_argument('--anno_file', type=str, default=None, help='annotation file contains the path of each video')
args = parser.parse_args()


i3d_debug = False


# Create thread pool
max_thread = 8 # there are some issues if too many threads
num_thread = args.num_thread if args.num_thread>0 and args.num_thread<=max_thread else max_thread
print(Fore.CYAN + 'thread #:', num_thread)
pool = ThreadPool(num_thread)
print('batch size: {}'.format(args.batch_size))

###### data path ######
if 'ucf' in args.data_path or 'hmdb' in args.data_path:
	path_input = args.data_path + '/'
else:
	path_input = args.data_path + args.video_in + '/'
feature_in_type = '.t7'

#--- create dataset folders
# path_output = args.feature_in + '_dbg_' + args.base_model + '/' 
path_output = args.feature_in + '_' + args.base_model + '/' 
if args.structure != 'tsn':
	path_output = args.feature_in + '-' + args.structure + '/'
if not os.path.isdir(path_output):
	os.makedirs(path_output)

###### set up the model ######
# Load the pretrained model
print(Fore.GREEN + 'Pre-trained model:', args.base_model)

if args.base_model == 'c3d':
	from C3D_model import C3D
	clip_size = 16
	model = C3D()
	model.load_state_dict(torch.load(args.pretrain_weight))
    
	list_model = list(model.children())
	list_conv = list_model[:-6]
	list_fc = list_model[-6:-4]
	extractor_conv = nn.Sequential(*list_conv)
	extractor_fc = nn.Sequential(*list_fc)

	# multi-gpu
	extractor_conv = torch.nn.DataParallel(extractor_conv.cuda())
	extractor_conv.eval()
	extractor_fc = torch.nn.DataParallel(extractor_fc.cuda())
	extractor_fc.eval()
elif args.base_model == 'i3d':
	import pytorch_i3d as i3d
	clip_size = 16
	model = i3d.InceptionI3d()
	# pdb.set_trace()
	# tmp = model.state_dict()['Mixed_4d.b2b.conv3d.weight'].clone(	
	model.load_state_dict(torch.load('/net/ca-home1/home/ma/grv/code/pytorch-i3d/models/rgb_imagenet.pt'))
	
	# pdb.set_trace()
	list_model = list(model.children())
	list_conv = list_model[3:]
	list_fc = list_model[:1]

	# not sure this way works
	extractor_conv = nn.Sequential(*list_conv)
	extractor_fc = nn.Sequential(*list_fc)

	extractor_conv = torch.nn.DataParallel(extractor_conv.cuda())
	extractor_conv.eval()
	extractor_fc = torch.nn.DataParallel(extractor_fc.cuda())
	extractor_fc.eval()
# elif args.base_model == 'i3d':
# 	import pytorch_i3d as i3d
# 	clip_size = 16
# 	model = i3d.InceptionI3d(feat_extractor=True)
# 	# pdb.set_trace()
# 	# tmp = model.state_dict()['Mixed_4d.b2b.conv3d.weight'].clone(	
# 	model.load_state_dict(torch.load('/net/ca-home1/home/ma/grv/code/pytorch-i3d/models/rgb_imagenet.pt'))
# 	model = torch.nn.DataParallel(model.cuda())
# 	model.eval()
else:
	model = getattr(models, args.base_model)(pretrained=True)
	# remove the last layer
	feature_map = list(model.children())
	feature_map.pop()
	extractor = nn.Sequential(*feature_map)
	# multi-gpu
	extractor = torch.nn.DataParallel(extractor.cuda())
	extractor.eval()

cudnn.benchmark = True

#--- data pre-processing
if args.base_model == 'c3d':
	data_transform = transforms.Compose([
		transforms.Resize(112),
		transforms.CenterCrop(112),
		transforms.ToTensor(),
		])
elif args.base_model == 'i3d':
	# data_transform = transforms.Compose([
	# 	VT.Resize(256), 
	# 	VT.RandomCrop(224), 
	# 	VT.RandomHorizontalFlip()
	# 	])
	data_transform = transforms.Compose([
		transforms.Resize(256), 
		transforms.CenterCrop(224),
		# SHOULD NOT RUN THIS for I3D
		# transforms.ToTensor()
		])
else:
	data_transform = transforms.Compose([
		transforms.Resize(224),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

################### Main Function ###################
def im2tensor(im):
	im = Image.fromarray(im) # convert numpy array to PIL image
	t_im = data_transform(im) # Create a PyTorch Variable with the transformed image
	# if the backbone is i3d, normalize to [-1, 1]
	
	if args.base_model == 'i3d':
		t_im = (np.array(t_im)/255.)*2 - 1
		t_im = torch.Tensor(t_im) # -> HxWxC
		t_im = t_im.permute([2,0,1])
		
	return t_im

def extract_frame_feature_batch(list_tensor):
	with torch.no_grad():		
		batch_tensor_list = convert_i3d_tensor_batch(list_tensor) # e.g. 113x3x16x224x224
		features_list = []
		for batch_tensor in batch_tensor_list:
			batch_tensor = Variable(batch_tensor).cuda() # Create a PyTorch Variable
			
			# pdb.set_trace()
			# # tmp  = model(batch_tensor[:2]) 
			# features_conv = extractor_conv(batch_tensor[:2])
			# tmp = extractor_fc(features_conv)
			# prev = torch.load('/net/acadia9a/data/jchoi/data/ucf_hmdb_full/TA3N/ucf101/RGB-feature2_i3d/Fencing/v_Fencing_g07_c01/img_00001.t7')
			# np.linalg.norm((tmp[0].squeeze().cpu()-prev).numpy()) 

			# features = model.extract_features(batch_tensor)
			features_conv = extractor_conv(batch_tensor)
			cur_features = extractor_fc(features_conv)
			features_list.append(cur_features)

		features = torch.cat(features_list, dim=0)
		features = features.view(features.size(0), -1).cpu()
		return features

def convert_c3d_tensor_batch(batch_tensor): # e.g. 30x3x112x112 --> 15x3x16x112x112
	batch_tensor_c3d = torch.Tensor()
	for b in range(batch_tensor.size(0)-clip_size+1):
		tensor_c3d = batch_tensor[b:b+clip_size,:,:,:]
		tensor_c3d = torch.transpose(tensor_c3d,0,1).unsqueeze(0)
		batch_tensor_c3d = torch.cat((batch_tensor_c3d, tensor_c3d))

	batch_tensor_c3d = batch_tensor_c3d*255
	return batch_tensor_c3d

def convert_i3d_tensor_batch(batch_tensor): 
	# Dense sampling (1 frame stride) of the input tensor to make a batch of clip tensors
	# e.g. 1x3xTx224x224 --> Bx3x16x224x224
	batch_tensor_i3d = torch.Tensor()
	for b in range(batch_tensor.size(0)-clip_size):
		tensor_i3d = batch_tensor[b:b+clip_size,:,:,:]
		tensor_i3d = torch.transpose(tensor_i3d,0,1).unsqueeze(0)
		batch_tensor_i3d = torch.cat((batch_tensor_i3d, tensor_i3d))
	
	num_chunks = int(np.ceil(batch_tensor_i3d.shape[0]/args.batch_size))
	batch_tensor_i3d_list = []
	for i in range(num_chunks-1):
		cur_tensor = batch_tensor_i3d[i*args.batch_size:(i+1)*args.batch_size]
		batch_tensor_i3d_list.append(cur_tensor)
	if (num_chunks-1)*args.batch_size < batch_tensor_i3d.shape[0]:
		cur_tensor = batch_tensor_i3d[(num_chunks-1)*args.batch_size:]
		batch_tensor_i3d_list.append(cur_tensor)

	return batch_tensor_i3d_list

def extract_features(video_file):
	print(video_file)
	if args.input_type == 'video':  # create the video folder if the data structure 
		video_name = os.path.splitext(video_file)[0]
	else:
		video_name = '_'.join(video_file.split('/')[-4:])  
	if args.structure == 'tsn':  # create the video folder if the data structure is TSN
		if not os.path.isdir(path_output + video_name + '/'):
			os.makedirs(path_output + video_name + '/')

	num_exist_files = len(os.listdir(path_output + video_name + '/'))

	frames_tensor = []
	# print(class_name)
	if args.input_type == 'video':
		tmp = video_file.split('/')
		
		# the frame read is RGB
		if 'ucf' in args.data_path or 'hmdb' in args.data_path:
			reader = imageio.get_reader(path_input + '/' + tmp[0] + '/' + tmp[1]+'.avi')
		else:
			reader = imageio.get_reader(path_input + '/' + tmp[1] + '/' + tmp[2])
		# reader = imageio.get_reader(path_input + class_name + '/' + video_file)

		#--- collect list of frame tensors
		try:
			for t, im in enumerate(reader):
				if np.sum(im.shape) != 0:
					id_frame = t+1
					frames_tensor.append(im2tensor(im))  # include data pre-processing
		except RuntimeError:
			print(Back.RED + 'Could not read frame', id_frame+1, 'from', video_file)
	elif args.input_type == 'frames':
		list_frames = os.listdir(video_file)
		list_frames.sort()

		# --- collect list of frame tensors
		try:
			for t in range(len(list_frames)):
				im = imageio.imread(video_file + '/' + list_frames[t])
				if np.sum(im.shape) != 0:
					id_frame = t+1
					frames_tensor.append(im2tensor(im))  # include data pre-processing
		except RuntimeError:
			print(Back.RED + 'Could not read frame', id_frame+1, 'from', video_file)

	#--- divide the list into two parts: major (can de divided by batch size) & the rest (will add dummy tensors)
	num_frames = len(frames_tensor)
	if num_frames == num_exist_files: # skip if the features are already saved
		print('Features already exist: {}'.format(path_output + video_name))
		return

	# zeropad the beginning and the end of the video 
	zero_padding = torch.zeros_like(torch.stack(frames_tensor)[:int(clip_size/2)])
	frames_batch = torch.cat([zero_padding, torch.stack(frames_tensor), zero_padding], dim=0)
	# frames_batch = F.pad(torch.stack(frames_tensor), pad=(0,0,0,0,0,0,int(clip_size/2),int(clip_size/2)), mode='constant', value=0)

	# num_major = num_frames//args.batch_size*args.batch_size
	# num_rest = num_frames - num_major

	# # add dummy tensor to make total size == batch_size*N
	# num_dummy = args.batch_size - num_rest
	# for i in range(num_dummy):
	# 	frames_tensor.append(torch.zeros_like(frames_tensor[0]))

	#--- extract video features
	# features = torch.Tensor()

	# for t in range(0, num_frames+num_dummy, args.batch_size):
	# 	frames_batch = frames_tensor[t:t+args.batch_size]
	# 	features_batch = extract_frame_feature_batch(frames_batch)
	# 	features = torch.cat((features,features_batch))

	# features = features[:num_frames] # remove the dummy part


	if i3d_debug:
		writer = imageio.get_writer('/net/acadia9a/data/jchoi/data/ucf_hmdb_full/TA3N/UCF/dbg/test.mp4', fps=25)
		for frame in frames_batch:
			# pdb.set_trace()
			writer.append_data(frame.permute([1,2,0]).numpy().astype(np.uint8))
		writer.close()
		pdb.set_trace()


	features_batch = extract_frame_feature_batch(frames_batch)

	assert features_batch.shape[0] == num_frames

	features = features_batch

	# features = features_batch[int(clip_size/2):int(features_batch.shape[0]-clip_size/2)] # remove the dummy part

	#--- save the frame-level feature vectors to files
	for t in range(features.size(0)):
		id_frame = t+1
		id_frame_name = str(id_frame).zfill(5)
		if args.structure == 'tsn':
			# pdb.set_trace()
			filename = path_output + video_name + '/' + 'img_' + id_frame_name + feature_in_type
		elif args.structure == 'imagenet':
			filename = path_output + class_name + '/' + video_name + '_' + id_frame_name + feature_in_type
		else:
			raise NameError(Back.RED + 'not valid data structure')

		if not os.path.exists(filename):
			torch.save(features[t].clone(), filename) # if no clone(), the size of features[t] will be the same as features

################### Main Program ###################
# parse the classes
if args.anno_file is not None:
    df = pd.read_csv(args.anno_file, header=None)
    data = df.to_numpy()

class_names = []
data_dict_org = {}
all_classes = []
for row in data:
	if 'ucf' in args.data_path or 'hmdb' in args.data_path:
		cur_cls = row[0].split('/')[0]
	elif 'kinetics' in args.data_path:
		cur_cls = row[0].split('/')[1]
	else:
		cur_cls = row[0].split('/')[-3]
	# class_names.append()
	if cur_cls not in data_dict_org:
		data_dict_org[cur_cls] = []
	data_dict_org[cur_cls].append(row[0])
	if cur_cls not in all_classes:
		all_classes.append(cur_cls)

# class_names_proc = np.unique(np.array(class_names))

# filter out classes
data_dict = {}
if  args.end_class == -1:
	selected_classes = all_classes[args.start_class:]
else:
	selected_classes = all_classes[args.start_class:args.end_class+1]
for k,v in data_dict_org.items():
	if k in selected_classes:
		data_dict[k] = v

# pdb.set_trace()

start = time.time()
for class_name, v in data_dict.items():
	if args.total_shards == 1:
		val = v
	else:
		v = v[args.shard_start_idx:]
		num_per_shard = int(np.floor(len(v)/args.total_shards))
		if args.cur_shard == args.total_shards-1:
			val = v[num_per_shard*args.cur_shard:]
		else:
			val = v[num_per_shard*args.cur_shard:num_per_shard*(args.cur_shard+1)]
	start_class = time.time()
	for cur_val in val:
		start_vid = time.time()
		extract_features(cur_val)
		# pdb.set_trace()
		# pool.map(extract_features, val, chunksize=1)
		end_vid = time.time()
		print('Elapsed time for ' + cur_val + ': ' + str(end_vid-start_vid))
	end_class = time.time()
	print('Elapsed time for ' + class_name + ': ' + str(end_class-start_class))

end = time.time()
print('Total elapsed time: ' + str(end-start))
print(Fore.GREEN + 'All the features are generated for ' + args.video_in)


# start = time.time()
# for class_name, val in data_dict.items():
# 	start_class = time.time()
# 	# extract_features(val[0])

# 	# pdb.set_trace()
# 	pool.map(extract_features, val, chunksize=1)

# 	end_class = time.time()
# 	print('Elapsed time for ' + class_name + ': ' + str(end_class-start_class))

# end = time.time()
# print('Total elapsed time: ' + str(end-start))
# print(Fore.GREEN + 'All the features are generated for ' + args.video_in)
