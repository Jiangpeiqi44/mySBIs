# import dlib
from skimage import io
# from skimage import transform as sktransform
import numpy as np
# from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from DeepFakeMask import dfl_full, facehull, components, extended
import albumentations as alb
import cv2
# import tqdm
import pywt
from oragn_mask import get_five_key, mask_patch

def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark


def name_resolve(path):
    name = os.path.splitext(os.path.basename(path))[0]
    vid_id, frame_id = name.split('_')[0:2]
    return vid_id, frame_id


def total_euclidean_distance(a, b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b, axis=1))


def random_get_hull(landmark, img1):
    hull_type = random.choice([0, 1, 2, 3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype(
            'int32'), face=img1, channels=3).mask
        return mask/255


def random_erode_dilate(mask, ksize=None):
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1)/255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):

    maskIndices = np.where(mask != 0)
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)
    maskPts = np.hstack(
        (maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))

    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    # print(faceSize.shape)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(
            hull, (int(maskPts[i, 0]), int(maskPts[i, 1])), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = \
        weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + \
        (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
        1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)

    maskIndices = np.where(mask != 0)

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class BIOnlineGeneration():
    def __init__(self):

        with open('src/utils/library/ff_lm.json', 'r') as f:
            self.landmarks_record = json.load(f)
            self.data_list = []
            for k, v in self.landmarks_record.items():
                self.landmarks_record[k] = np.array(v)
                self.data_list.append(k)

        # extract all frame from all video in the name of {videoid}_{frameid}
        self.source_transforms = self.get_source_transforms()
        # self.base_path = 'H:/Academic/ustc_face_forgery/Dataset/FF++/original_sequences/youtube/raw/frames/'
        self.base_path = 'data/FF++/original_sequences/youtube/raw/frames/'
        self.stats = 'None'  # ['BI','SBI']
        self.ibi_data_list = []
        self.not_aug_flag = False
        # predefine mask distortion
        self.distortion = iaa.Sequential(
            [iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        self.elastic = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

    def gen_one_datapoint(self, background_face_path=None, landmark_bi=None, phase=None):
        if background_face_path == None:
            background_face_path = random.choice(self.data_list)
        self.this_landmark = landmark_bi
        if background_face_path != None:
            self.phase = phase

        face_img, mask_bi, mask = self.get_blended_face(background_face_path)
        if self.not_aug_flag:
            mask = (1 - mask) * mask * 4

        return face_img, mask_bi, mask

    def get_blended_face(self, background_face_path):
        background_face = io.imread(background_face_path)
        # Image.fromarray(np.uint8(background_face)).convert('RGB').save("imgs/background_face.png")
        # ## 完成背景脸的crop
        background_face, background_landmark, __, ___ = crop_face(
            background_face, self.this_landmark, margin=True, crop_by_bbox=False)
        # ## 搜索最近前脸
        foreground_face_path = self.search_similar_face(
            background_landmark, background_face_path)
        # ## 完成前脸crop
        vid_idx, frame_idx = name_resolve(foreground_face_path)
        foreground_face_path = os.path.join(
            self.base_path, '{}/{}.png'.format(vid_idx, frame_idx))
        
        foreground_face = io.imread(foreground_face_path)
        foreground_landmark_abs = reorder_landmark(np.load(
            foreground_face_path.replace('/frames/', '/landmarks/').replace('png', 'npy'))[0])
        foreground_face, _, __, ___ = crop_face(
            foreground_face, foreground_landmark_abs, margin=True, crop_by_bbox=False)

        # ## get random type of initial blending mask
        if np.random.rand() < 0.25:
            background_landmark = background_landmark[:68]
        
        # 全脸Mask
        if True:
        # if np.random.rand() < 0.5:
            mask = random_get_hull(background_landmark, background_face)

            # ## random deform mask
            mask = self.elastic(image=mask)['image']
            mask = random_erode_dilate(mask)
            mask_bi = mask.copy()
            # ## filte empty mask after deformation
            if np.sum(mask) == 0:
                raise NotImplementedError

            self.not_aug_flag = False  # False
            isDownScale = False  # False
            isBIBlend = False  # False
            blur_flag = True  # True

            if np.random.rand() < 0.5:
                isDownScale = True
                if np.random.rand() < 0.25:
                    isBIBlend = True

            if isDownScale:
                    h, w, c = background_face.shape
                    ori_size = (w, h)
                    size_down = random.randint(128, 317)
                    aug_size = (size_down, size_down)
                    background_face = cv2.resize(
                        background_face, aug_size, interpolation=cv2.INTER_LINEAR).astype('uint8')
                    foreground_face = cv2.resize(
                        foreground_face, aug_size, interpolation=cv2.INTER_LINEAR).astype('uint8')
                    mask = cv2.resize(
                        mask, aug_size, interpolation=cv2.INTER_LINEAR).astype('float32')

            # ## apply color transfer
            if self.stats == 'BI':
                foreground_face = colorTransfer(
                    background_face, foreground_face, mask*255)
            elif self.stats == 'IBI':
                foreground_face = colorTransfer(
                    background_face, foreground_face, mask*255)
                if np.random.rand() < 0.5:
                    self.not_aug_flag = True
                if np.random.rand() < 0.5:
                    blur_flag = False

            # ## 添加STG 如果是IBI有概率触发不增强，仅保留混合边界
            if not self.not_aug_flag:
                if np.random.rand() < 0.5:
                    foreground_face = self.source_transforms(
                        image=foreground_face.astype(np.uint8))['image']
                else:
                    background_face = self.source_transforms(
                        image=background_face.astype(np.uint8))['image']
            # ## blend two face  小波 or 默认方法

            if isBIBlend:
                blended_face, mask = blendImages(
                    foreground_face, background_face, mask*255)
            else:
                # blended_face, mask = wavelet_blend(
                #     foreground_face, background_face, mask[:, :, 0])
                if self.not_aug_flag:
                    if np.random.rand() < 0.5:
                    # if True:
                        blended_face, mask = dynamic_blend(
                            foreground_face, background_face, mask[:, :, 0], 1, blur_flag=blur_flag)
                    else:
                        blended_face, mask = dynamic_blend_align(
                            foreground_face, background_face, mask[:, :, 0], 1, blur_flag=blur_flag)
                else:
                    if np.random.rand() < 0.5:
                    # if True:
                        blended_face, mask = dynamic_blend(
                            foreground_face, background_face, mask[:, :, 0])
                    else:
                        blended_face, mask = dynamic_blend_align(
                            foreground_face, background_face, mask[:, :, 0])
            # ## resize back to default resolution
            if isDownScale:
                blended_face = cv2.resize(
                    blended_face, ori_size, interpolation=cv2.INTER_LINEAR).astype('uint8')
                mask = cv2.resize(
                    mask, ori_size, interpolation=cv2.INTER_LINEAR).astype('float32')
                if not isBIBlend:
                    mask = mask.reshape(mask.shape+(1,))

            blended_face = blended_face.astype(np.uint8)

            mask = mask[:, :, 0:1]
            mask_bi = mask_bi[:, :, 0:1]
        # 五官区域Mask
        else:
            five_key = get_five_key(background_landmark)
            reg = np.random.randint(0, 10)
            # reg = 6 # 只换嘴部
            # 得到deform后的mask
            mask, mask_bi = mask_patch(reg, background_face, five_key)
            # ##随机对源或目标进行变换
            foreground_face = colorTransfer(
                    background_face, foreground_face, mask_bi*255)
            if np.random.rand() < 0.5:
                foreground_face = self.source_transforms(
                        image=foreground_face.astype(np.uint8))['image']
            else:
                background_face = self.source_transforms(
                        image=background_face.astype(np.uint8))['image']
            # 直接混合
            # if True:
            if np.random.rand() < 0.5:
                blended_face, _ = dynamic_blend(foreground_face, background_face, mask[:,:,0], blur_flag=False)
            # blended_face, mask = blendImages(foreground_face, background_face, mask_bi*255)
            else:
                blended_face, _ = dynamic_blend_align(foreground_face, background_face, mask[:,:,0], blur_flag=False)
                
        return blended_face, mask_bi, mask

    def search_similar_face(self, this_landmark, background_face_path):
        # windows下的反斜杠逆天bug
        vid_id = background_face_path.split('/')[-2]
        min_dist = 99999999
        if self.stats == 'BI':
            # random sample 5000 frame from all frams:
            all_candidate_path = random.sample(self.data_list, k=2500)

            # filter all frame that comes from the same video as background face
            all_candidate_path = filter(lambda k: name_resolve(k)[
                                        0] != vid_id, all_candidate_path)

            all_candidate_path = list(all_candidate_path)
        elif self.stats == 'IBI':
            all_candidate_path = self.ibi_data_list
            all_candidate_path = filter(
                lambda k: k != background_face_path, all_candidate_path)
            all_candidate_path = list(all_candidate_path)
            all_candidate_path = random.sample(self.ibi_data_list, k=5)
            all_candidate_path = ['{}_{}'.format(
                vid_id, os.path.basename(i)) for i in all_candidate_path]
        else:
            raise NotImplementedError
        # loop throungh all candidates frame to get best match
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(
                np.float32)
            candidate_distance = total_euclidean_distance(
                candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path
        return min_path

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
                # 添加额外的增强
                # alb.RandomToneCurve (scale=0.01, p=0.1),
                # alb.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='train'):
    assert phase in ['train', 'val', 'test']

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1-x0
        h = y1-y0
        w0_margin = w/4  # 0#np.random.rand()*(w/8)
        w1_margin = w/4
        h0_margin = h/4  # 0#np.random.rand()*(h/5)
        h1_margin = h/4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1-x0
        h = y1-y0
        w0_margin = w/8  # 0#np.random.rand()*(w/8)
        w1_margin = w/8
        h0_margin = h/2  # 0#np.random.rand()*(h/5)
        h1_margin = h/5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        w1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h0_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
        h1_margin *= (np.random.rand()*0.6+0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0-h0_margin))
    y1_new = min(H, int(y1+h1_margin)+1)
    x0_new = max(0, int(x0-w0_margin))
    x1_new = min(W, int(x1+w1_margin)+1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p-x0_new, q-y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p-x0_new, q-y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W/r), int(H/r)),
                            interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds


def dynamic_blend(source, target, mask, blend_ratio=None, blur_flag=True):
    if blur_flag:
        mask_blured = get_blend_mask(mask)
    else:
        mask_blured = mask.reshape((mask.shape+(1,)))
    if source.shape != target.shape:
        h, w, c = target.shape
        source = cv2.resize(
            source, (w, h), interpolation=cv2.INTER_LINEAR).astype('uint8')
    if blend_ratio == None:
        blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    if blur_flag:
        mask_blured_ret = mask_blured
    else:
        mask_blured_ret = get_blend_mask(mask)*blend_ratio
    return img_blended, mask_blured_ret


def dynamic_blend_align(source, target, mask, blend_ratio=None, blur_flag=True):
    # source 前景  target背景 mask与target对应
    slice_flag = False
    if source.shape != target.shape:
        # 这里进行对齐，而不是直接reshape
        h1, w1, _ = target.shape
        h2, w2, _ = source.shape
        h_max, w_max = max(h1, h2), max(w1, w2)
        delta_s_h = max(h_max - h2, 0)
        delta_s_w = max(w_max - w2, 0)
        delta_t_h = max(h_max - h1, 0)
        delta_t_w = max(w_max - w1, 0)
        pad_mask = np.pad(mask, ((0, delta_t_h), (0, delta_t_w)), 'constant')
        pad_source = np.pad(
            source, ((0, delta_s_h), (0, delta_s_w), (0, 0)), 'constant')
        pad_target = np.pad(
            target, ((0, delta_t_h), (0, delta_t_w), (0, 0)), 'constant')
        # print(pad_mask.shape,pad_source.shape,pad_target.shape)
        mask = pad_mask
        source = pad_source
        target = pad_target
        if np.random.rand() < 0.5:
            slice_flag = True
    if blur_flag:
        mask_blured = get_blend_mask(mask)
    else:
        mask_blured = mask.reshape((mask.shape+(1,)))
    if blend_ratio == None:
        blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    if blur_flag:
        mask_blured_ret = mask_blured
    else:
        mask_blured_ret = get_blend_mask(mask)*blend_ratio
    if slice_flag:
        img_blended = img_blended[0:h1, 0:w1, :]
        mask_blured_ret = mask_blured_ret[0:h1, 0:w1, :]
    return img_blended, mask_blured_ret


def wavelet_blend(source, target, mask, wavelet_type='bior3.3'):
    # H W C(RGB)
    mask_blured = get_blend_mask(mask)
    h, w, c = target.shape
    # blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
    source = cv2.resize(
        source, (w, h), interpolation=cv2.INTER_LINEAR).astype('uint8')
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
    # print(f'{blend_ratio_A:.4f}', f'{blend_ratio_H:.4f}', f'{blend_ratio_V:.4f}', f'{blend_ratio_D:.4f}')
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


if __name__ == '__main__':
    ds = BIOnlineGeneration()
    img, mask, label = ds.gen_one_datapoint()
    # from tqdm import tqdm
    # all_imgs = []
    # for _ in tqdm(range(50)):
    #     img,mask,label = ds.gen_one_datapoint()
    #     mask = np.repeat(mask,3,2)
    #     mask = (mask*255).astype(np.uint8)
    #     img_cat = np.concatenate([img,mask],1)
    #     all_imgs.append(img_cat)
    # all_in_one = Image.new('RGB', (2570,2570))

    # for x in range(5):
    #     for y in range(10):
    #         idx = x*10+y
    #         im = Image.fromarray(all_imgs[idx])

    #         dx = x*514
    #         dy = y*257

    #         all_in_one.paste(im, (dx,dy))

    # all_in_one.save("all_in_one.jpg")
