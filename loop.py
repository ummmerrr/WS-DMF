# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-9#
import os,glob,numbers
# 图像处理
import math,cv2,random
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from grad import GradUtil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

#start#
import time, tqdm
from torchvision.transforms import functional as f

class KerasBackend(object):
	bests = {'auc':0, 'iou':0, 'f1s':0, 'a':0}

	path_minlos = 'checkpoint_minloss.pt'
	path_metric = 'checkpoint_metrics.tar'
	paths = dict()
	logTxt = []
	isParallel = False
	def __init__(self, args, **kargs):
		super(KerasBackend, self).__init__()
		self.args = args
		# print('*'*32,'device')
		torch.manual_seed(311)
		self.device = torch.device('cpu')
		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')  
			torch.cuda.empty_cache()
			torch.cuda.manual_seed_all(311)
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.enabled = True
			# Benchmark模式会提升计算速度，但计算中随机性使得每次网络前馈结果略有差异，
			# deterministic避免这种波动, 设置为False可以牺牲GPU提升精度

			current_device = torch.cuda.current_device()
			print(self.device, torch.cuda.get_device_name(current_device))
			for i in range(torch.cuda.device_count()):
				print("    {}:".format(i), torch.cuda.get_device_name(i))
		
	def save_weights(self, path):
		if not os.path.exists(self.root):
			os.mkdir(self.root)
		if self.isParallel:
			torch.save(self.model.module.state_dict(), path)
		else:
			torch.save(self.model.state_dict(), path)
		# print('save weigts to path:{}'.format(path))
	
	def load_weights(self, mode, desc=True):
		path = self.paths.get(mode, mode)#返回完全路径或者mode
		if mode=='los':
			path = self.path_minlos
		try:
			pt = torch.load(path, map_location=self.device)
			self.model.load_state_dict(pt, strict=False)#
			if self.isParallel:
				self.model = self.model.module
			if desc:print('Load from:', path)
			return True
		except Exception as e:
			print('Load wrong:', path)
			return False

	def init_weights(self):
		print('*'*32, 'Initial Weights--Ing!')
		for m in self.model.modules():
			if isinstance(m, nn.Conv2d) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.Linear) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.BatchNorm2d) and  m.weight.requires_grad:
				torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
				torch.nn.init.constant_(m.bias.data, 0.0)

	def init_folders(self, dataset, losStr):
		if self.args.root=='':
			self.root = 'EXP-WS-DMF'
			if self.args.bug:
				self.root = 'BUG'+self.root
		else:
			self.root = self.args.root

		dataset.expCross = hasattr(dataset, 'expCross') and dataset.expCross
		if dataset.expCross: 
			self.path_metric = '{}/{}xcp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}xlos.pt'.format(self.root, dataset.dbname)
		else:
			self.path_metric = '{}/{}_cp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}_los.pt'.format(self.root, dataset.dbname)
		print('Folder for experiment:', self.root)

		name_pt = dataset.dbname+'x' if dataset.expCross else dataset.dbname
		# print('Exec:', self.root)
		for key in self.bests.keys():
			self.paths[key] = '{}/{}-{}.pt'.format(self.root, name_pt, key)

	def compile(self, dataset, loss='fr', lr=0.01, **args): 
		#设置路径
		self.dataset = dataset
		self.init_folders(dataset, ''.join(loss))

		# 参数设置：反向传播、断点训练 
		self.gradUtil = GradUtil(model=self.model, loss=loss, lr=lr, root=self.root)
		if not self.load_weights(self.path_minlos):
			self.init_weights()
		print('Params total(KB):',sum(p.numel() for p in self.model.parameters()))#//245
		print('Params train(KB):',sum(p.numel() for p in self.model.parameters() if p.requires_grad))
			
		if self.isParallel:
			print('*'*32, 'Model Parallel')
			self.model = nn.DataParallel(self.model)  #, device_ids=[0,1,2,3]
			# self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))  
			# torch.cuda.set_device(self.device)
			# self.device = torch.device('cuda:0')
			self.model.to(self.device)
		else:
			print('*'*32, 'Model Serial')
			self.model = self.model.to(self.device) 

		try:
			self.bests = torch.load(self.path_metric)
			print('Metric Check point:', self.bests)
		except:
			print('Metric Check point none!') 
		
		self.gradUtil.criterion = self.gradUtil.criterion.to(self.device)
		
	def callBackModelCheckPoint(self, scores, lossItem=1e9):
		logStr = '\t'
		for mode in scores.keys():
			if scores[mode]>self.bests[mode]:
				logStr += '{}:{:6.4f}->{:6.4f},'.format(mode, self.bests[mode], scores[mode])
				self.bests[mode] = scores[mode]
				self.save_weights(self.paths[mode])   
		print(logStr)
		self.logTxt.append(logStr)
		torch.save(self.bests, self.path_metric)
		
	stop_counter=0
	stop_training = False
	best_loss = 9999
	isBestLoss = False
	def callBackEarlyStopping(self, los, epoch=0, patience=18):
		if los<self.best_loss:
			print('\tlos={:6.4f}->{:6.4f}'.format(self.best_loss, los))
			self.best_loss = los
			self.stop_counter=0
			# self.save_weights(self.path_minlos)
			self.isBestLoss = True
		else:
			print('\tlos={:6.4f}'.format(los))
			self.stop_counter+=1
			if self.stop_counter>patience and self.gradUtil.isLrLowest(1e-5) and epoch>169:
				self.stop_training = True
				print('EarlyStopp after:', patience)
	
		if self.isBestLoss:
			self.isBestLoss = False
			self.save_weights(self.path_minlos)

class KerasTorch(KerasBackend):
	evalEpochs = 3
	evalMetric = True
	evalEpochs=3
	def __init__(self, model, **kargs):
		super(KerasTorch, self).__init__(**kargs)
		self.model = model

	def desc(self, key='my'):#, self.scheduler.get_lr()[0] 
		# print('Learing Rate:', self.optimizer.param_groups[0]['lr'])
		for n,m in self.model.named_parameters():
			if n.__contains__(key):
				print(n,m.detach().cpu().numpy())

	def fit(self, epochs=196):#现行验证，意义不大，把所有权重都验证要花不少时间
		self.stop_counter = 0
		self.stop_training = False            
		print('\n', '*'*32,'fitting:'+self.root) 
		# self.desc()
		time_fit_begin = time.time()
		for i in range(epochs):
			time_stamp = time.time()

			# 训练
			lossItem = self.train()

			logStr = '{:03}$ los={}'.format(i, lossItem)
			print('\r'+logStr)
			self.logTxt.append(logStr)
			self.gradUtil.update_scheduler(i)
			# 验证
			if self.evalMetric and self.gradUtil.isLrLowest(thresh=1e-4):
				# if i>1 and i%self.evalEpochs==0:
				scores, lossItem = self.val()
				logStr = '{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f}'
				logStr = logStr.format(i, scores['auc'],scores['iou'],scores['f1s'])
				print('\r'+logStr)
				self.logTxt.append(logStr)
				self.callBackEarlyStopping(lossItem, i)
				self.callBackModelCheckPoint(scores)

			else:
				self.callBackEarlyStopping(lossItem)#eye does not use this line
			# 早停
			if self.stop_training==True:
				print('Stop Training!!!')
				break
			
			time_epoch = time.time() - time_stamp
			print('{:03}* {:.2f} mins, left {:.2f} hours to run'.format(i, time_epoch/60, time_epoch/60/60*(epochs-i)))
			if self.args.bug and i>2:
				break
		self.desc()
		if self.evalMetric:
			print(self.bests)
		self.logTxt.append(str(self.bests))
		with open(self.root + '/logs.txt', 'w') as f:
			f.write('\n'.join(self.logTxt))

		logTime = '\nRunning {:.2f} hours for {} epochs!'.format((time.time() - time_fit_begin)/60/60, epochs)
		print(logTime)

	def train(self):
		torch.set_grad_enabled(True)
		self.model.train()     
		lossItem = 0
		tbar = tqdm.tqdm(self.dataset.trainSet(bs=self.args.bs))
		for i, imgs in enumerate(tbar):
			(img, lab, fov, aux) = self.dataset.parse(imgs)#cpu
			lab = lab.to(self.device)
			fov = fov.to(self.device)
			aux = aux.to(self.device)
			if not (isinstance(img, dict) or isinstance(img, list)):
				img = img.to(self.device)

			losInit = []
			# print(img.shape)
			out = self.model(img)

			if self.args.coff_rot!=0:
				los = self.model.fcn.regular_rot() * self.args.coff_rot
				losInit.append(los)

			# print('backward:', out.shape, lab.shape, fov.shape)
			_lossItem, losStr = self.gradUtil.backward_seg(out, lab, fov, self.model, requires_grad=True, losInit=losInit) 
			lossItem += _lossItem
			del out, lab, fov, aux   
			# print('\r{:03}$ los={:.3f}'.format(i, _lossItem), end='')
			tbar.set_description('{:03}$ {:.3f}={}'.format(i, _lossItem, losStr))
			if self.args.bug and i>3:
				break
		return lossItem

	def predict(self, img, *args):
		self.model.eval()
		torch.set_grad_enabled(False)
		# with torch.no_grad():  
		if not (isinstance(img, dict) or isinstance(img, list)):
			img = img.to(self.device)      
		pred = self.model(img)#*fov.to(self.device)
		if isinstance(pred, dict):
			pred = pred['pred']
		if isinstance(pred, (list, tuple)):
			pred = pred[0]
		pred = pred.detach()
		# pred = pred*fov if fov is not None else pred
		return pred.clamp(0, 1)



	def val(self):
		torch.set_grad_enabled(False)
		self.model.eval()        
		sum_auc = 0
		sum_iou = 0
		sum_f1s = 0
		sum_los = 0

		dataloader = self.dataset.valSet()
		for i, imgs in enumerate(dataloader):
			(img, lab, fov, aux) = self.dataset.parse(imgs) 
			pred = self.predict(img)
			loss = self.gradUtil.backward_seg(pred, lab.to(self.device), fov, self.model, requires_grad=False)[0] 
			sum_los += loss
			true = lab.squeeze().numpy().astype(np.float32)
			pred = pred.cpu().squeeze().numpy().astype(np.float32)

			true = true.reshape(-1)
			pred = pred.reshape(-1)
			if fov is not None:
				fov = fov.view(-1).numpy().astype(np.bool_)  
				true, pred = true[fov], pred[fov]
				
			true = np.round(true)
			auc = metrics.roc_auc_score(true, pred)
			sum_auc += auc

			pred = np.round(np.clip(pred, 1e-6, 1-1e-6))
			iou = metrics.jaccard_score(true, pred)
			sum_iou += iou
			f1s = metrics.f1_score(true, pred, average='binary')
			sum_f1s += f1s
			print('\r{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f} & a={:.4f}'.format(i, auc, iou, f1s, f1s), end='')
			if self.args.bug and i>2:
				break
		num = len(dataloader)#i+1#
		los = sum_los/num
		scores = {'auc':sum_auc/num, 'iou':sum_iou/num, 'f1s':sum_f1s/num, 'a':sum_f1s/num}
		return scores, los
#end#
