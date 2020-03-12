# import argparse
import os
import time
import shutil
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset import TSNDataSet
from models_dbg import VideoModel
from loss import *
from opts import parser
from utils.utils import randSelectBatch
import math

from colorama import init
from colorama import Fore, Back, Style
import numpy as np
from tensorboardX import SummaryWriter
import pdb

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

best_prec1 = 0
gpu_count = torch.cuda.device_count()

def main():
	global args, best_prec1, writer
	args = parser.parse_args()

	print(Fore.GREEN + 'Baseline:', args.baseline_type)
	print(Fore.GREEN + 'Frame aggregation method:', args.frame_aggregation)

	print(Fore.GREEN + 'target data usage:', args.use_target)
	if args.use_target == 'none':
		print(Fore.GREEN + 'no Domain Adaptation')
	else:
		if args.dis_DA != 'none':
			print(Fore.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
			if len(args.place_dis) != args.add_fc + 2:
				raise ValueError(Back.RED + 'len(place_dis) should be equal to add_fc + 2')

		if args.adv_DA != 'none':
			print(Fore.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

		if args.use_bn != 'none':
			print(Fore.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

	# determine the categories
	class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
	num_class = len(class_names)

	# pdb.set_trace()

	#=== check the folder existence ===#
	path_exp = args.exp_path + args.modality + '/'
	if not os.path.isdir(path_exp):
		os.makedirs(path_exp)

	if args.tensorboard:
		writer = SummaryWriter(path_exp + '/tensorboard')  # for tensorboardX

	#=== initialize the model ===#
	print(Fore.CYAN + 'preparing the model......')
	model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
				train_segments=args.num_segments, val_segments=args.val_segments, 
				base_model=args.arch, path_pretrained=args.pretrained,
				add_fc=args.add_fc, fc_dim = args.fc_dim,
				dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
				use_bn=args.use_bn if args.use_target != 'none' else 'none', ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
				n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
				use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
				verbose=args.verbose, share_params=args.share_params)

	model = torch.nn.DataParallel(model, args.gpus).cuda()

	if args.optimizer == 'SGD':
		print(Fore.YELLOW + 'using SGD')
		optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optimizer == 'Adam':
		print(Fore.YELLOW + 'using Adam')
		optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
	else:
		print(Back.RED + 'optimizer not support or specified!!!')
		exit()

	#=== check point ===#
	start_epoch = 1
	print(Fore.CYAN + 'checking the checkpoint......')
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch'] + 1
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch'])))
			if args.resume_hp:
				print("=> loaded checkpoint hyper-parameters")
				optimizer.load_state_dict(checkpoint['optimizer'])
		else:
			print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	#--- open log files ---#
	if not args.evaluate:
		if args.resume:
			train_file = open(path_exp + 'train.log', 'a')
			train_short_file = open(path_exp + 'train_short.log', 'a')
			val_file = open(path_exp + 'val.log', 'a')
			val_short_file = open(path_exp + 'val_short.log', 'a')
			train_file.write('========== start: ' + str(start_epoch) + '\n')  # separation line
			train_short_file.write('========== start: ' + str(start_epoch) + '\n')
			val_file.write('========== start: ' + str(start_epoch) + '\n')
			val_short_file.write('========== start: ' + str(start_epoch) + '\n')
		else:
			train_short_file = open(path_exp + 'train_short.log', 'w')
			val_short_file = open(path_exp + 'val_short.log', 'w')
			train_file = open(path_exp + 'train.log', 'w')
			val_file = open(path_exp + 'val.log', 'w')

		val_best_file = open(args.save_best_log, 'a')

	else:
		test_short_file = open(path_exp + 'test_short.log', 'w')
		test_file = open(path_exp + 'test.log', 'w')

	#=== Data loading ===#
	print(Fore.CYAN + 'loading data......')

	if args.use_opencv:
		print("use opencv functions")

	if args.modality == 'RGB':
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus']:
		data_length = 5

	# calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
	num_source = sum(1 for i in open(args.train_source_list))
	num_target = sum(1 for i in open(args.train_target_list))
	num_val = sum(1 for i in open(args.val_list))

	num_iter_source = num_source / args.batch_size[0]
	num_iter_target = num_target / args.batch_size[1]
	num_max_iter = max(num_iter_source, num_iter_target)
	num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
	num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

	# calculate the weight for each class
	class_id_list = [int(line.strip().split(' ')[2]) for line in open(args.train_source_list)]
	class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
	class_freq = (class_data_counts / class_data_counts.sum()).tolist()

	weight_source_class = torch.ones(num_class).cuda()
	weight_domain_loss = torch.Tensor([1, 1]).cuda()

	if args.weighted_class_loss == 'Y':
		weight_source_class = 1 / torch.Tensor(class_freq).cuda()

	if args.weighted_class_loss_DA == 'Y':
		weight_domain_loss = torch.Tensor([1/num_source_train, 1/num_target_train]).cuda()

	# data loading (always need to load the testing data)
	val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
	val_set = TSNDataSet("", args.val_list, num_dataload=num_val, num_segments=val_segments,
						 new_length=data_length, modality=args.modality,
						 image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
																		  "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
						 random_shift=False,
						 test_mode=True,
						 )
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size[2], shuffle=False,
											 num_workers=args.workers, pin_memory=True)

	if not args.evaluate:
		source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
		source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

		target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target_train, num_segments=args.num_segments,
								new_length=data_length, modality=args.modality,
								image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
								random_shift=False,
								test_mode=True,
								)

		target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
		target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)

	# --- Optimizer ---#
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		criterion = torch.nn.CrossEntropyLoss(weight=weight_source_class).cuda()
		criterion_domain = torch.nn.CrossEntropyLoss(weight=weight_domain_loss).cuda()
	else:
		raise ValueError("Unknown loss type")

	if args.evaluate:
		print(Fore.CYAN + 'evaluation only......')
		prec1 = validate(val_loader, model, criterion, num_class, 0, test_file)
		test_short_file.write('%.3f\n' % prec1)
		return

	#=== Training ===#
	start_train = time.time()
	print(Fore.CYAN + 'start training......')
	beta = args.beta
	gamma = args.gamma
	mu = args.mu
	loss_c_current = 999 # random large number
	loss_c_previous = 999 # random large number

	attn_source_all = torch.Tensor()
	attn_target_all = torch.Tensor()

	for epoch in range(start_epoch, args.epochs+1):

		## schedule for parameters
		alpha = 2 / (1 + math.exp(-1 * (epoch) / args.epochs)) - 1 if args.alpha < 0 else args.alpha

		## schedule for learning rate
		if args.lr_adaptive == 'loss':
			adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
		elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
			adjust_learning_rate(optimizer, args.lr_decay)

		# train for one epoch
		loss_c = train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, train_file, train_short_file, alpha, beta, gamma, mu)
		
		# update the recorded loss_c
		loss_c_previous = loss_c_current
		loss_c_current = loss_c

		# evaluate on validation set
		if epoch % args.eval_freq == 0 or epoch == args.epochs:
			prec1 = validate(val_loader, model, criterion, num_class, epoch, val_file)

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			line_update = ' ==> updating the best accuracy' if is_best else ''
			line_best = "Best score {} vs current score {}".format(best_prec1, prec1) + line_update
			print(Fore.YELLOW + line_best)
			val_short_file.write('%.3f\n' % prec1)

			best_prec1 = max(prec1, best_prec1)

			if args.tensorboard:
				writer.add_text('Best_Accuracy', str(best_prec1), epoch)

			if args.save_model:
				save_checkpoint({
					'epoch': epoch,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict(),
					'best_prec1': best_prec1,
					'prec1': prec1,
				}, is_best, path_exp)
	
	end_train = time.time()
	print(Fore.CYAN + 'total training time:', end_train - start_train)
	val_best_file.write('%.3f\n' % best_prec1)

	# --- write the total time to log files ---#
	line_time = 'total time: {:.3f} '.format(end_train - start_train)
	if not args.evaluate:
		train_file.write(line_time)
		train_short_file.write(line_time)
		val_file.write(line_time)
		val_short_file.write(line_time)
	else:
		test_file.write(line_time)
		test_short_file.write(line_time)

	#--- close log files ---#
	if not args.evaluate:
		train_file.close()
		train_short_file.close()
		val_file.close()
		val_short_file.close()
	else:
		test_file.close()
		test_short_file.close()

	if args.tensorboard:
		writer.close()

	if args.save_attention >= 0:
		np.savetxt('attn_source_' + str(args.save_attention) + '.log', attn_source_all.cpu().detach().numpy(), fmt="%s")
		np.savetxt('attn_target_' + str(args.save_attention) + '.log', attn_target_all.cpu().detach().numpy(), fmt="%s")


def train(num_class, source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log, log_short, alpha, beta, gamma, mu):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses_a = AverageMeter()  # adversarial loss
	losses_d = AverageMeter()  # discrepancy loss
	losses_e = AverageMeter()  # entropy loss
	losses_s = AverageMeter()  # ensemble loss
	losses_c = AverageMeter()  # classification loss
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	data_loader = enumerate(zip(source_loader, target_loader))

	# step info
	start_steps = epoch * len(source_loader)
	total_steps = args.epochs * len(source_loader)

	attn_epoch_source = torch.Tensor()
	attn_epoch_target = torch.Tensor()
	for i, ((source_data, source_label),(target_data, target_label)) in data_loader:
		# pdb.set_trace()
		source_data = source_data[:,1,:]
		# debugging with dummy data
		# source_data = source_data.repeat([1,1,2]) 
		# target_data = target_data.repeat([1,1,2]) 
		# pdb.set_trace()
		# debugging with dummy data

		# setup hyperparameters
		p = float(i + start_steps) / total_steps
		beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
		beta = [beta_dann if beta[i] < 0 else beta[i] for i in range(len(beta))] # replace the default beta if value < 0

		source_size_ori = source_data.size()  # original shape
		target_size_ori = target_data.size()  # original shape
		batch_source_ori = source_size_ori[0]
		batch_target_ori = target_size_ori[0]
		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_source_ori < args.batch_size[0]:
			source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1])
			source_data = torch.cat((source_data, source_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if source_data.size(0) % gpu_count != 0:
			source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1))
			source_data = torch.cat((source_data, source_data_dummy))

		# measure data loading time
		data_time.update(time.time() - end)

		source_label = source_label.cuda(non_blocking=True) # pytorch 0.4.X
		target_label = target_label.cuda(non_blocking=True) # pytorch 0.4.X

		if args.baseline_type == 'frame':
			source_label_frame = source_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames
			target_label_frame = target_label.unsqueeze(1).repeat(1, args.num_segments).view(-1)

		label_source = source_label_frame if args.baseline_type == 'frame' else source_label  # determine the label for calculating the loss function
		label_target = target_label_frame if args.baseline_type == 'frame' else target_label

		#------ forward pass data again ------#
		out_source = model(source_data)

		# ignore dummy tensors
		out_source = out_source[:batch_source_ori]

		#------ calculate the loss function ------#
		# 1. calculate the classification loss
		out = out_source
		label = label_source

		# pdb.set_trace()
		loss = criterion(out, label)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient and args.verbose:
				print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

		optimizer.step()
		
		pred = out

		prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))

		losses.update(loss.item())
		top1.update(prec1.item(), out_source.size(0))
		top5.update(prec5.item(), out_source.size(0))
		losses_c.update(loss.item(), out_source.size(0)) # pytorch 0.4.X

		if i % args.print_freq == 0:
			line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
				   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
				   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
				   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
				   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' + \
				   'Loss {loss.val:.4f} ({loss.avg:.4f})   loss_c {loss_c.avg:.4f}\t'

			line = line.format(
				epoch, i, len(source_loader), batch_time=batch_time, data_time=data_time, alpha=alpha, beta=beta, gamma=gamma, mu=mu,
				loss=losses, loss_c=losses_c, top1=top1, top5=top5,
				lr=optimizer.param_groups[0]['lr'])

			if i % args.show_freq == 0:
				print(line)

			log.write('%s\n' % line)

	return losses_c.avg

def validate(val_loader, model, criterion, num_class, epoch, log):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()

	# initialize the embedding
	if args.tensorboard:
		feat_val_display = None
		label_val_display = None

	for i, (val_data, val_label) in enumerate(val_loader):
		# pdb.set_trace()

		# debug
		# val_data = val_data.repeat([1,1,2])
		# debug 

		val_size_ori = val_data.size()  # original shape
		batch_val_ori = val_size_ori[0]

		# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
		if batch_val_ori < args.batch_size[2]:
			val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
			val_data = torch.cat((val_data, val_data_dummy))

		# add dummy tensors to make sure batch size can be divided by gpu #
		if val_data.size(0) % gpu_count != 0:
			val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1), val_data.size(2))
			val_data = torch.cat((val_data, val_data_dummy))

		val_label = val_label.cuda(non_blocking=True)
		with torch.no_grad():

			if args.baseline_type == 'frame':
				val_label_frame = val_label.unsqueeze(1).repeat(1,args.num_segments).view(-1) # expand the size for all the frames

			# compute output
			out_val = model(val_data[:,1,:])

			# ignore dummy tensors
			out_val = removeDummy(out_val, batch_val_ori)

			# measure accuracy and record loss
			label = val_label_frame if args.baseline_type == 'frame' else val_label

			pred = out_val

			if args.baseline_type == 'tsn':
				pred = pred.view(val_label.size(0), -1, num_class).mean(dim=1) # average all the segments (needed when num_segments != val_segments)

			loss = criterion(pred, label)
			prec1, prec5 = accuracy(pred.data, label, topk=(1, 5))

			losses.update(loss.item(), out_val.size(0))
			top1.update(prec1.item(), out_val.size(0))
			top5.update(prec5.item(), out_val.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				line = 'Test: [{0}][{1}/{2}]\t' + \
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' + \
					  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'

				line = line.format(
					   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
					   top1=top1, top5=top5)

				if i % args.show_freq == 0:
					print(line)

				log.write('%s\n' % line)

	if args.tensorboard:  # update the embedding every iteration
		# embedding
		n_iter_val = epoch * len(val_loader)

		writer.add_embedding(feat_val_display, metadata=label_val_display.data, global_step=n_iter_val, tag='validation')

	print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses)))

	return top1.avg


def save_checkpoint(state, is_best, path_exp, filename='checkpoint.pth.tar'):

	path_file = path_exp + filename
	torch.save(state, path_file)
	if is_best:
		path_best = path_exp + 'model_best.pth.tar'
		shutil.copyfile(path_file, path_best)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, decay):
	"""Sets the learning rate to the initial LR decayed by 10 """
	for param_group in optimizer.param_groups:
		param_group['lr'] /= decay

def adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
	ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y), '<=': (lambda x, y: x <= y)}
	if ops[op](stat_current, stat_previous):
		for param_group in optimizer.param_groups:
			param_group['lr'] /= decay

def adjust_learning_rate_dann(optimizer, p):
	for param_group in optimizer.param_groups:
		param_group['lr'] = args.lr / (1. + 10 * p) ** 0.75

def loss_adaptive_weight(loss, pred):
	weight = 1 / pred.var().log()
	constant = pred.std().log()
	return loss * weight + constant

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

# remove dummy tensors
def removeDummy(out_1, batch_size):
	out_1 = out_1[:batch_size]
	return out_1

if __name__ == '__main__':
	main()
