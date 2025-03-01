# -*- encoding:utf-8 -*-
#start#
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
import socket
import matplotlib as mpl
if 'TAN' not in socket.gethostname():
	print('Run on Server!!!')
	mpl.use('Agg')#服务器绘图

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def gain(ret, p=1):    #gain_off
	mean = np.mean(ret)
	ret_min = mean-(mean-np.min(ret))*p
	ret_max = mean+(np.max(ret)-mean)*p
	ret = 255*(ret - ret_min)/(ret_max - ret_min)
	ret = np.clip(ret, 0, 255).astype(np.uint8)
	return ret

def arr2img(pic):
	return Image.fromarray(pic.astype(np.uint8))#, mode='L'

def arrs2imgs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = arr2img(pic[key])
	return _pic

def imgs2arrs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = np.array(pic[key])
	return _pic

def pil_tran(pic, tran=None):
	if tran is None:
		return pic
	if isinstance(tran, list):
		for t in tran:
			for key in pic.keys():
				pic[key] = pic[key].transpose(t)
	else:
		for key in pic.keys():
			pic[key] = pic[key].transpose(tran)
	return pic

class Aug4Val(object):
	number = 8
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4Val.number
		if flag==0:
			return pic
		pic = arrs2imgs(pic)
		if flag==1:
			return imgs2arrs(pil_tran(pic, tran=Image.Transpose.FLIP_LEFT_RIGHT))
		if flag==2:
			return imgs2arrs(pil_tran(pic, tran=Image.Transpose.FLIP_TOP_BOTTOM))
		if flag==3:
			return imgs2arrs(pil_tran(pic, tran=Image.Transpose.ROTATE_180))
		if flag==4:
			return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE]))
		if flag==5:
			return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.Transpose.FLIP_TOP_BOTTOM]))
		if flag==6:
			return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.Transpose.FLIP_LEFT_RIGHT]))
		if flag==7:
			return imgs2arrs(pil_tran(pic, tran=[Image.TRANSPOSE,Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]))


class EyeSetResource(object):
	size = dict()
	def __init__(self, folder='../eyeset', dbname='drive', loo=None, **args):
		super(EyeSetResource, self).__init__()
		
		self.folder = r'/kaggle/input/ws-dmf/WS-DMF-main/data/dataset/DRIVE'
			
		self.dbname = dbname

		self.imgs, self.labs, self.fovs, self.skes = self.getDataSet(self.dbname)
		if dbname=='stare' and loo is not None and loo<20: 
			self.imgs['test'] = [self.imgs['full'][loo]]
			self.imgs['train'] = self.imgs['full'][:loo] + self.imgs['full'][1+loo:]
			self.imgs['val'] = self.imgs['train']
			
			self.labs['test'] = [self.labs['full'][loo]]
			self.labs['train'] = self.labs['full'][:loo] + self.labs['full'][1+loo:]
			self.labs['val'] = self.labs['train']
			
			self.fovs['test'] = [self.fovs['full'][loo]]
			self.fovs['train'] = self.fovs['full'][:loo] + self.fovs['full'][1+loo:]
			self.fovs['val'] = self.fovs['train']
			
			self.skes['test'] = [self.skes['full'][loo]]
			self.skes['train'] = self.skes['full'][:loo] + self.skes['full'][1+loo:]
			self.skes['val'] = self.skes['train']
			print('LOO:', loo, self.imgs['test'])
			print('LOO:', loo, self.labs['test'])
			print('LOO:', loo, self.fovs['test'])
			print('LOO:', loo, self.skes['test'])

		self.lens = {'train':len(self.labs['train']),   'val':len(self.labs['val']),
					 'test':len(self.labs['test']),     'full':len(self.labs['full'])}  
		# print(self.lens)  
		if self.lens['test']>0:
			lab = self.readArr(self.labs['test'][0])
			self.size['raw'] = lab.shape
			h,w = lab.shape
			self.size['pad'] = (math.ceil(h/32)*32, math.ceil(w/32)*32)
			print('size:', self.size)
		else:
			print('dataset has no images!')

		# print('*'*32,'eyeset','*'*32)
		strNum = 'images:{}+{}+{}#{}'.format(self.lens['train'], self.lens['val'], self.lens['test'], self.lens['full'])
		print('{}@{}'.format(self.dbname, strNum))

	def getDataSet(self, dbname):        
		# 测试集
		imgs_test = self.readFolder(dbname, part='test', image='rgb')
		labs_test = self.readFolder(dbname, part='test', image='lab')
		fovs_test = self.readFolder(dbname, part='test', image='fov')
		skes_test = self.readFolder(dbname, part='test', image='ske')
		# 训练集
		imgs_train = self.readFolder(dbname, part='train', image='rgb')
		labs_train = self.readFolder(dbname, part='train', image='lab')
		fovs_train = self.readFolder(dbname, part='train', image='fov')
		skes_train = self.readFolder(dbname, part='train', image='ske')
		# 全集
		imgs_full,labs_full,fovs_full,skes_full = [],[],[],[]
		imgs_full.extend(imgs_train); imgs_full.extend(imgs_test)
		labs_full.extend(labs_train); labs_full.extend(labs_test)
		fovs_full.extend(fovs_train); fovs_full.extend(fovs_test)
		skes_full.extend(skes_train); skes_full.extend(skes_test)

		db_imgs = {'train': imgs_train, 'val':imgs_train, 'test': imgs_test, 'full':imgs_full}
		db_labs = {'train': labs_train, 'val':labs_train, 'test': labs_test, 'full':labs_full}
		db_fovs = {'train': fovs_train, 'val':fovs_train, 'test': fovs_test, 'full':fovs_full}
		db_skes = {'train': skes_train, 'val':skes_train, 'test': skes_test, 'full':skes_full}
		return db_imgs, db_labs, db_fovs, db_skes

	def readFolder(self, dbname, part='train', image='rgb'):
		path = self.folder + '/' + dbname + '/' + part + '_' + image
		imgs = glob.glob(path + '/*.npy')
		imgs.sort()
		return imgs
		
	def readArr(self, image):
		# assert(image.endswith('.npy'), 'not npy file!') 
		return np.load(image) 
	
	def readDict(self, index, exeData):  
		img = self.readArr(self.imgs[exeData][index])
		fov = self.readArr(self.fovs[exeData][index])
		lab = self.readArr(self.labs[exeData][index])
		ske = self.readArr(self.skes[exeData][index])
		if fov.shape[-1]==3:
			fov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
		return {'img':img, 'lab':lab, 'fov':fov, 'ske':ske}#

import imgaug as ia
import imgaug.augmenters as iaa
IAA_NOISE = iaa.OneOf(children=[# Noise
		iaa.Add((-7, 7), per_channel=True),
		iaa.AddElementwise((-7, 7)),
		iaa.Multiply((0.9, 1.1), per_channel=True),
		iaa.MultiplyElementwise((0.9, 1.1), per_channel=True),

		iaa.AdditiveGaussianNoise(scale=3, per_channel=True),
		iaa.AdditiveLaplaceNoise(scale=3, per_channel=True),
		iaa.AdditivePoissonNoise(lam=5, per_channel=True),

		iaa.SaltAndPepper(0.01, per_channel=True),
		iaa.ImpulseNoise(0.01),
	]
)

TRANS_NOISE = iaa.Sequential(children=[IAA_NOISE])

from albumentations import (
	# 空间
	RGBShift, ChannelDropout, ChannelShuffle, 
	# 色调
	HueSaturationValue, RandomContrast, RandomBrightness, 
	# 翻转
	Flip, Transpose, RandomRotate90, PadIfNeeded, RandomGridShuffle,
	# 变形
	GridDistortion, ShiftScaleRotate, IAAPiecewiseAffine, OpticalDistortion, ElasticTransform,#IAAPerspective, 
	# 噪声
	IAASharpen, IAAEmboss, GaussNoise, MultiplicativeNoise, #IAAAdditiveGaussianNoise, 
	# 模糊
	MedianBlur, GaussianBlur, #Blur, MotionBlur,
	# 其他
	OneOf, Compose, CropNonEmptyMaskIfExists, CLAHE, RandomGamma
) # 图像变换函数

TRANS_TEST = Compose([CLAHE(p=1), RandomGamma(p=1)])#
TRANS_AAUG = Compose([       
	OneOf([Transpose(p=1), RandomRotate90(p=1), ], p=.7),
	Flip(p=.7), 
])


from skimage import morphology
from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':8, 'val':Aug4Val.number, 'test':1}
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full

	SIZE_IMAGE = 128
	expCross = False   
	LEN_AUG = 32
	def __init__(self, datasize=128, **args):
		super(EyeSetGenerator, self).__init__(**args)
		self.SIZE_IMAGE = datasize
		self.LEN_AUG = 96 // (datasize//64)**2
		print('SIZE_IMAGE:{} & AUG SIZE:{}'.format(self.SIZE_IMAGE, self.LEN_AUG))
		
	def __len__(self):
		length = self.lens[self.exeData]*self.exeNums[self.exeMode]
		if self.isTrainMode:
			return length*self.LEN_AUG
		return length

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'full' if self.expCross else mode 
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	def trainSet(self, bs=8, data='train'):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, data='val'):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	def testSet(self, data='test'):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦
	
	# @staticmethod
	def parse(self, pics, cat=True):
		rows, cols = pics['lab'].squeeze().shape[-2:]     
		for key in pics.keys(): 
			# print(key, pics[key].shape)
			pics[key] = pics[key].view(-1,1,rows,cols) 
		return pics['img'], torch.round(pics['lab']), torch.round(pics['fov']), torch.round(pics['ske'])

	def post(self, img, lab, fov):
		if type(img) is not np.ndarray:img = img.squeeze().cpu().numpy()
		if type(lab) is not np.ndarray:lab = lab.squeeze().cpu().numpy()
		if type(fov) is not np.ndarray:fov = fov.squeeze().cpu().numpy()
		img = img * fov
		return img, lab, fov

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	use_csm = True
	def __getitem__(self, idx, divide=32):
		index = idx % self.lens[self.exeData] 
		pics = self.readDict(index, self.exeData)
		imag = pics['img']  # original image
		
		if self.isTrainMode:
			# Ensure lab, fov, and ske have the same shape
			lab_shape = pics['lab'].shape
			if pics['fov'].shape != lab_shape:
				pics['fov'] = cv2.resize(pics['fov'], (lab_shape[1], lab_shape[0]), interpolation=cv2.INTER_NEAREST)
			if pics['ske'].shape != lab_shape:
				pics['ske'] = cv2.resize(pics['ske'], (lab_shape[1], lab_shape[0]), interpolation=cv2.INTER_NEAREST)
				
			mask = np.stack([pics['lab'], pics['fov'], pics['ske']], axis=-1)
			# Crop augmentation based on non-empty mask
			augCrop = CropNonEmptyMaskIfExists(p=1, height=self.SIZE_IMAGE, width=self.SIZE_IMAGE)
			picaug = augCrop(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']

			imag = TRANS_TEST(image=imag)['image']

			# Add noise
			imag = TRANS_NOISE(image=imag)
			# Additional augmentation
			picaug = TRANS_AAUG(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']
			
			pics['img'] = imag
			pics['lab'], pics['fov'], pics['ske'] = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
		else:
			pics['img'] = TRANS_TEST(image=pics['img'])['image']
			h, w = pics['lab'].shape
			w = int(np.ceil(w / divide)) * divide
			h = int(np.ceil(h / divide)) * divide
			augPad = PadIfNeeded(p=1, min_height=h, min_width=w)
			for key in pics.keys():
				pics[key] = augPad(image=pics[key])['image']

			if self.isValMode:
				flag = idx // self.lens[self.exeData]
				pics = Aug4Val.forward(pics, flag)

		if pics['img'].shape[-1] == 3:
			pics['img'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)

		for key in pics.keys():
			pics[key] = torch.from_numpy(pics[key]).type(torch.float32).div(255)
		return pics

#end#




if __name__ == '__main__':
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='drive', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='hrf', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='chase', isBasedPatch=True)#
	# /kaggle/input/ws-dmf/WS-DMF-main/data/dataset/DRIVE
	db = EyeSetGenerator(folder=r'/kaggle/input/ws-dmf/WS-DMF-main/data/dataset/DRIVE', dbname='stare', loo=0)#
	# db = EyeSetGenerator(folder=r'G:\Objects\expSeg\datasets\seteye', dbname='drive', isBasedPatch=False)#
	# db.expCross = True
	print('generator:', len(db.trainSet()), len(db.valSet()), len(db.testSet()), )

	# db.expCross=True
	for i, imgs in enumerate(db.trainSet(1)):
	# for i, imgs in enumerate(db.valSet(1)):
	# for i, imgs in enumerate(db.testSet()):
		# print(imgs.keys())
		# print(imgs)
		(img, lab, fov, aux) = db.parse(imgs)
		print(img.shape, lab.shape, fov.shape, aux.shape)

		plt.subplot(221),plt.imshow(tensor2image(img))
		plt.subplot(222),plt.imshow(tensor2image(lab))
		plt.subplot(223),plt.imshow(tensor2image(fov))
		plt.subplot(224),plt.imshow(tensor2image(aux))
		plt.show()
