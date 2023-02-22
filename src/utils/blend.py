# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import cv2
import numpy as np
import scipy as sp
from skimage.measure import label, regionprops
import random
from PIL import Image
import sys
import pywt

def custom_blend(source, target, mask):
	mask_blured = get_blend_mask(mask)
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_x_ray = mask_blured.copy()
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	# print(blend_ratio)
	return img_blended, mask_blured, mask_x_ray


def custom_blend_opt(source, target, mask):
	mask_blured = get_blend_mask(mask)
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_x_ray = mask_blured.copy()
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	# print(blend_ratio)
	return img_blended, mask_blured, mask_x_ray, blend_ratio


def custom_blend_3(source, target, mask):
	mask_blured = get_blend_mask(mask)
	mask_x_ray = mask_blured.copy()
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	# print(blend_ratio)
	return img_blended, mask_blured, mask_x_ray


def custom_blend_2(source, target, mask, mask1):
	mask_blured = get_alpha_blend_mask(mask)
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	blend_list_2 = [0, 0.25, 0.25, 0.25, 0.5, 0.75]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_x_ray = mask_blured.copy()
 	###
	# ratio_0 = 0.5
	# blend_ratio = 0.75
	ratio_0 = blend_list_2[np.random.randint(len(blend_list_2))]
	mask_blured *= blend_ratio
	mask_blured = mask_blured + ratio_0 * \
		(1 - blend_ratio) * (mask1.reshape(mask1.shape+(1,)))
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	# print(blend_ratio)
	return img_blended, mask_blured, mask_x_ray, blend_ratio, ratio_0


def custom_blend_with_rand_mask(source, target, mask):
	mask_blured = get_custom_blend_mask(mask)  # get_alpha_blend_mask(mask) #
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_x_ray = mask_blured.copy()
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	# print(blend_ratio)
	return img_blended, mask_blured, mask_x_ray


def alpha_blend(source, target, mask):
	# ? get alpha blend mask ?
	mask_blured = get_blend_mask(mask)
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured


def alpha_blend_2(source, target, mask):
	mask_blured = get_alpha_blend_mask(mask)
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured


def alpha_blend_3(source, target, mask):
	mask_blured = mask.reshape(mask.shape+(1,))
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured


def dynamic_blend(source, target, mask):
	mask_blured = get_blend_mask(mask)
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	blend_ratio = blend_list[np.random.randint(len(blend_list))]
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured

def dynamic_blend_DEBUG(source, target, mask, ratio):
	mask_blured = get_blend_mask(mask)
	
	blend_ratio = ratio
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured

def haar_blend(source, target, mask):
    # H W C(RGB)
	mask_blured = get_blend_mask(mask)
	# H, W ,C = mask_blured.shape
	
	blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
	# blend_ratio_A = blend_list[np.random.randint(len(blend_list))]
	# blend_ratio_H = blend_list[np.random.randint(len(blend_list))]
	# blend_ratio_V = blend_list[np.random.randint(len(blend_list))]
	# blend_ratio_D = blend_list[np.random.randint(len(blend_list))]
	# if np.random.rand() < 0.5:
	blend_ratio_A = 0.5 + np.random.rand()/2
	blend_ratio_H = 0.5 + np.random.rand()/2
	blend_ratio_V = 0.5 + np.random.rand()/2
	blend_ratio_D = 0.5 + np.random.rand()/2
	# blend_ratio_A = 0.5
	# blend_ratio_H = 1
	# blend_ratio_V = 1
	# blend_ratio_D = 1
	print(f'{blend_ratio_A:.4f}',f'{blend_ratio_H:.4f}',f'{blend_ratio_V:.4f}',f'{blend_ratio_D:.4f}')
	#  haar小波需要对每一个RGB通道
	source_haar = []
	target_haar = []
	for i in range(3):
		A, (H, V, D) =  pywt.dwt2(source[:,:,i], 'haar')
		dict_coeff = {}
		dict_coeff['A'] = A
		dict_coeff['H'] = H
		dict_coeff['V'] = V
		dict_coeff['D'] = D
		source_haar.append(dict_coeff)
	for i in range(3):
		A, (H, V, D) =  pywt.dwt2(target[:,:,i], 'haar')
		dict_coeff = {}
		dict_coeff['A'] = A
		dict_coeff['H'] = H
		dict_coeff['V'] = V
		dict_coeff['D'] = D
		target_haar.append(dict_coeff)
	H_haar, W_haar = source_haar[0]['A'].shape
	mask_blured_ori = mask_blured.copy()
	mask_blured = cv2.resize(mask_blured, (W_haar, H_haar), interpolation=cv2.INTER_AREA).astype('float32')
	# mask_blured = np.ones((H_haar, W_haar))
	# mask_blured = mask_blured.reshape(mask_blured.shape+(1,))
	# print(source_haar[0]['A'].shape, mask_blured.shape)
	# mask_blured = mask_blured[:,:,0] # 恢复 H_haar W_haar维度
	blend_coeff = []
	for i in range(3):
		dict_coeff = {}
		dict_coeff['A'] = source_haar[i]['A'] * mask_blured * blend_ratio_A + target_haar[i]['A'] * (1-(mask_blured*blend_ratio_A))
		dict_coeff['H'] = source_haar[i]['H'] * mask_blured * blend_ratio_H + target_haar[i]['H'] * (1-(mask_blured*blend_ratio_H))
		dict_coeff['V'] = source_haar[i]['V'] * mask_blured * blend_ratio_V + target_haar[i]['V'] * (1-(mask_blured*blend_ratio_V))
		dict_coeff['D'] = source_haar[i]['D'] * mask_blured * blend_ratio_D + target_haar[i]['D'] * (1-(mask_blured*blend_ratio_D))
		blend_coeff.append(dict_coeff)
	blend_RGB = []
	for i in range(3):
		img_one_ch = pywt.idwt2(
			(blend_coeff[i]['A'], (blend_coeff[i]['H'], blend_coeff[i]['V'], blend_coeff[i]['D'])), 'haar')
		blend_RGB.append(img_one_ch.reshape(img_one_ch.shape+(1,)))
	img_blended = np.concatenate((blend_RGB[0],blend_RGB[1],blend_RGB[2]),axis=2)
	return img_blended, mask_blured_ori


def wavelet_blend(source, target, mask, wavelet_type = 'bior3.5'):
    # H W C(RGB)
    # print(wavelet_type)
    mask_blured = get_blend_mask(mask)
    h, w, c = target.shape
    # blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
    # source = cv2.resize(
    #     source, (w, h), interpolation=cv2.INTER_LINEAR).astype('uint8')
    # blend_ratio_A = blend_list[np.random.randint(len(blend_list))]
    # blend_ratio_H = blend_list[np.random.randint(len(blend_list))]
    # blend_ratio_V = blend_list[np.random.randint(len(blend_list))]
    # blend_ratio_D = blend_list[np.random.randint(len(blend_list))]
    # if np.random.rand() < 0.5:
    blend_ratio_A = 0.5 + np.random.rand()/2
    blend_ratio_H = 0.5 + np.random.rand()/2
    blend_ratio_V = 0.5 + np.random.rand()/2
    blend_ratio_D = 0.5 + np.random.rand()/2
    # blend_ratio_A = 0
    # blend_ratio_H = 0
    # blend_ratio_V = 0
    # blend_ratio_D = 0
    print(f'{blend_ratio_A:.4f}', f'{blend_ratio_H:.4f}', f'{blend_ratio_V:.4f}', f'{blend_ratio_D:.4f}')
    #  wavelet小波需要对每一个RGB通道
    source_wavelet = []
    target_wavelet = []
      # 重构支撑范围为2Nr+1,分解支撑范围为2Nd+1。biorNr.Nd小波
    for i in range(3):
        A, (H, V, D) = pywt.dwt2(source[:, :, i], wavelet_type)
        dict_coeff = {}
        dict_coeff['A'] = A
        dict_coeff['H'] = H
        dict_coeff['V'] = V
        dict_coeff['D'] = D
        source_wavelet.append(dict_coeff)
    for i in range(3):
        A, (H, V, D) = pywt.dwt2(target[:, :, i], wavelet_type)
        dict_coeff = {}
        dict_coeff['A'] = A
        dict_coeff['H'] = H
        dict_coeff['V'] = V
        dict_coeff['D'] = D
        target_wavelet.append(dict_coeff)
    H_wavelet, W_wavelet = source_wavelet[0]['A'].shape
    mask_blured_ori = mask_blured.copy()
    mask_blured = cv2.resize(
        mask_blured, (W_wavelet, H_wavelet), interpolation=cv2.INTER_AREA).astype('float32')
    blend_coeff = []
    for i in range(3):
        dict_coeff = {}
        dict_coeff['A'] = source_wavelet[i]['A'] * mask_blured * \
            blend_ratio_A + target_wavelet[i]['A'] * \
            (1-(mask_blured*blend_ratio_A))
        dict_coeff['H'] = source_wavelet[i]['H'] * mask_blured * \
            blend_ratio_H + target_wavelet[i]['H'] * \
            (1-(mask_blured*blend_ratio_H))
        dict_coeff['V'] = source_wavelet[i]['V'] * mask_blured * \
            blend_ratio_V + target_wavelet[i]['V'] * \
            (1-(mask_blured*blend_ratio_V))
        dict_coeff['D'] = source_wavelet[i]['D'] * mask_blured * \
            blend_ratio_D + target_wavelet[i]['D'] * \
            (1-(mask_blured*blend_ratio_D))
        blend_coeff.append(dict_coeff)
    blend_RGB = []
    for i in range(3):
        img_one_ch = pywt.idwt2(
            (blend_coeff[i]['A'], (blend_coeff[i]['H'], blend_coeff[i]['V'], blend_coeff[i]['D'])), wavelet_type)
        blend_RGB.append(img_one_ch.reshape(img_one_ch.shape+(1,)))
    img_blended = np.concatenate(
        (blend_RGB[0], blend_RGB[1], blend_RGB[2]), axis=2)
    return img_blended, mask_blured_ori


def adv_blend(source, target, mask, blend_ratio):
	mask_blured = get_blend_mask(mask)
	mask_blured *= blend_ratio
	img_blended = (mask_blured * source + (1 - mask_blured) * target)
	return img_blended, mask_blured


def get_blend_mask(mask):
	H, W = mask.shape
	size_h = np.random.randint(192, 257)
	size_w = np.random.randint(192, 257)
	mask = cv2.resize(mask, (size_w, size_h))
	kernel_1 = random.randrange(5, 26, 2)
	kernel_1 = (kernel_1, kernel_1)
	kernel_2 = random.randrange(5, 26, 2)
	kernel_2 = (kernel_2, kernel_2)

	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured < 1] = 0

	mask_blured = cv2.GaussianBlur(
		mask_blured, kernel_2, np.random.randint(5, 46))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured, (W, H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_custom_blend_mask(mask):
	H, W = mask.shape
	size_h = np.random.randint(192, 257)
	size_w = np.random.randint(192, 257)
	mask = cv2.resize(mask, (size_w, size_h))
	kernel_1 = random.randrange(5, 10, 2)
	kernel_1 = (kernel_1, kernel_1)
	kernel_2 = random.randrange(5, 10, 2)
	kernel_2 = (kernel_2, kernel_2)
	# print(kernel_1,kernel_2)

	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured < 1] = 0

	mask_blured = cv2.GaussianBlur(
		mask_blured, kernel_2, np.random.randint(5, 13))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured, (W, H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_custom_blend_mask_small(mask):
	H, W = mask.shape
	size_h = np.random.randint(192, 257)
	size_w = np.random.randint(192, 257)
	mask = cv2.resize(mask, (size_w, size_h))
	kernel_1 = random.randrange(3, 12, 2)
	kernel_1 = (kernel_1, kernel_1)
	kernel_2 = random.randrange(3, 12, 2)
	kernel_2 = (kernel_2, kernel_2)
	# print(kernel_1,kernel_2)

	mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured[mask_blured < 1] = 0

	mask_blured = cv2.GaussianBlur(
		mask_blured, kernel_2, np.random.randint(3, 10))
	mask_blured = mask_blured/(mask_blured.max())
	mask_blured = cv2.resize(mask_blured, (W, H))
	return mask_blured.reshape((mask_blured.shape+(1,)))


def get_alpha_blend_mask(mask):
	kernel_list = [(11, 11), (9, 9), (7, 7), (5, 5), (3, 3)]
	blend_list = [0.25, 0.5, 0.75]
	kernel_idxs = random.choices(range(len(kernel_list)), k=2)
	blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
	mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
	# print(mask_blured.max())
	mask_blured[mask_blured < mask_blured.max()] = 0
	mask_blured[mask_blured > 0] = 1
	# mask_blured = mask
	mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
	mask_blured = mask_blured/(mask_blured.max())
	return mask_blured.reshape((mask_blured.shape+(1,)))
