# import dlib
from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from DeepFakeMask import dfl_full, facehull, components, extended
import albumentations as alb
import cv2
import tqdm


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
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(
            hull, (int(maskPts[i, 0]), int(maskPts[i, 1])), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0],
                                                                               maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

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
        # with open('precomuted_landmarks.json', 'r') as f:
        #     self.landmarks_record =  json.load(f)
        #     for k,v in self.landmarks_record.items():
        #         self.landmarks_record[k] = np.array(v)
        # print(self.landmarks_record)
        # extract all frame from all video in the name of {videoid}_{frameid}
        self.landmarks_record = {}
        self.data_list = [
            '000_0000.png',
            '001_0000.png'
        ]
        self.source_transforms = self.get_source_transforms()
        # predefine mask distortion
        self.distortion = iaa.Sequential(
            [iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        self.elastic = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

    def gen_one_datapoint(self, background_face_path=None):
        if background_face_path == None:
            background_face_path = random.choice(self.data_list)
        # data_type = 'real' if random.randint(0,1) else 'fake'
        data_type = 'fake'
        if data_type == 'fake':
            face_img, mask = self.get_blended_face(background_face_path)
            # mask = ( 1 - mask ) * mask * 4
        else:
            face_img = io.imread(background_face_path)
            mask = np.zeros((380, 380, 1))

        # randomly downsample after BI pipeline
        # if random.randint(0,1):
        #     aug_size = random.randint(64, 380)
        #     face_img = Image.fromarray(face_img)
        #     if random.randint(0,1):
        #         face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
        #     else:
        #         face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
        #     face_img = face_img.resize((380, 380),Image.BILINEAR)
        #     face_img = np.array(face_img)

        # random jpeg compression after BI pipeline
        # if random.randint(0,1):
        #     quality = random.randint(60, 100)
        #     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        #     face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
        #     face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)

        # face_img = face_img[60:380,30:287,:]
        # mask = mask[60:380,30:287,:]

        # random flip
        # if random.randint(0,1):
        #     face_img = np.flip(face_img,1)
        #     mask = np.flip(mask,1)

        return face_img, mask, data_type, background_face_path

    def get_blended_face(self, background_face_path):
        background_face = io.imread(background_face_path)
        # Image.fromarray(np.uint8(background_face)).convert('RGB').save("imgs/background_face.png")
        background_landmark = self.landmarks_record[background_face_path]
        foreground_face_path = self.search_similar_face(
            background_landmark, background_face_path)
        foreground_face = io.imread(foreground_face_path)

        # down sample before blending
        # aug_size = random.randint(128,380)
        # background_landmark = background_landmark * (aug_size/380)
        # foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        # background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        # Image.fromarray(np.uint8(background_face)).convert('RGB').save("imgs/background_face.png")
        # Image.fromarray(np.uint8(foreground_face)).convert('RGB').save("imgs/foreground_face.png")
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
        # Image.fromarray(np.uint8(mask*255)).convert('RGB').save("imgs/mask.png")
        #  random deform mask
        # mask = self.distortion.augment_image(mask)
        mask = self.elastic(image=mask)['image']
        mask = random_erode_dilate(mask)
        # Image.fromarray(np.uint8(mask*255)).convert('RGB').save("imgs/mask_distor.png")
        # filte empty mask after deformation
        if np.sum(mask) == 0:
            raise NotImplementedError

        # apply color transfer
        # foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        # 添加STG
        if np.random.rand() < 0.5:
            foreground_face = self.source_transforms(
                image=foreground_face.astype(np.uint8))['image']
        else:
            background_face = self.source_transforms(
                image=background_face.astype(np.uint8))['image']
        # blend two face
        blended_face, mask = blendImages(
            foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)

        # resize back to default resolution
        # blended_face = sktransform.resize(blended_face,(380,380),preserve_range=True).astype(np.uint8)
        # mask = sktransform.resize(mask,(380,380),preserve_range=True)
        mask = mask[:, :, 0:1]
        # Image.fromarray(np.uint8(mask[:,:,0]*255)).convert('RGB').save("imgs/mask.png")
        return blended_face, mask

    def search_similar_face(self, this_landmark, background_face_path):
        # vid_id, frame_id = name_resolve(background_face_path)
        min_dist = 99999999

        # # random sample 5000 frame from all frams:
        # all_candidate_path = random.sample( self.data_list, k=5000)

        # # filter all frame that comes from the same video as background face
        # all_candidate_path = filter(lambda k:name_resolve(k)[0] != vid_id, all_candidate_path)
        # all_candidate_path = list(all_candidate_path)
        all_candidate_path = self.data_list

        all_candidate_path = filter(
            lambda k: k != background_face_path, all_candidate_path)
        all_candidate_path = list(all_candidate_path)

        # loop throungh all candidates frame to get best match
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(
                np.float32)
            candidate_distance = total_euclidean_distance(
                candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path
        # print(min_path)
        return min_path

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)


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
