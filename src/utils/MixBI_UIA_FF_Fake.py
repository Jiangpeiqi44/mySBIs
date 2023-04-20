# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE
import torch.nn.functional as F
import logging
import math
import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, IterableDataset
from scipy.ndimage import binary_erosion, binary_dilation
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import traceback

warnings.filterwarnings('ignore')

# win version ?
if os.path.isfile('src/utils/library/vit_gen_pcl.py'):
    sys.path.append(
        'src/utils/library/')
    print('exist library')
    exist_bi = True
else:
    exist_bi = False

def init_ff_fake(type, level='frame', n_frames=8):
    # raw or c23 or c40
    dataset_path = 'data/FF++/manipulated_sequences/{}/raw/frames/'.format(type)

    image_list = []
    label_list = []

    folder_list = sorted(glob(dataset_path+'*'))
   
    if level == 'video':
        label_list = [0]*len(folder_list)
        return folder_list, label_list
    for i in range(len(folder_list)):
        # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
        images_temp = sorted(glob((folder_list[i]+'/*.png')))
        # images_temp = [os.path.join(folder_list[i], k).replace('\\', '/')
                #    for k in sorted(os.listdir(folder_list[i]))]
        if n_frames < len(images_temp):
            images_temp = [images_temp[round(i)] for i in np.linspace(
                0, len(images_temp)-1, n_frames)]
        image_list += images_temp
        label_list += [1]*len(images_temp)

    return image_list, label_list

class FF_Dataset(Dataset):
    def __init__(self, phase='train', image_size=224, n_frames=8):

        assert phase in ['train', 'val', 'test']
        image_list=[]
        label_list=[]
        types = ['Deepfakes','Face2Face','Faceswap','NeuralTextures']
        for type in types:
            image_list_one, label_list_one = init_ff_fake(type, 'frame', n_frames=n_frames)
            image_list = image_list + image_list_one
            label_list = label_list + label_list_one
        path_lm = '/landmarks/'
        label_list = [label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace(
            '/frames/', path_lm).replace('.png', '.npy')) and os.path.isfile(image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        image_list = [image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace(
            '/frames/', path_lm).replace('.png', '.npy')) and os.path.isfile(image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        self.path_lm = path_lm
        print(f'FF({phase}): {len(image_list)}')

        self.image_list = image_list
        self.image_size = (image_size, image_size)
        self.phase = phase
        self.n_frames = n_frames
        self.transforms = self.get_transforms()
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                
                img = np.array(Image.open(filename))
                landmark = np.load(filename.replace(
                    '.png', '.npy').replace('/frames/', self.path_lm))[0]
                bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(
                ), landmark[:, 0].max(), landmark[:, 1].max()])
                bboxes = np.load(filename.replace(
                    '.png', '.npy').replace('/frames/', '/retina/'))[:2]
                iou_max = -1
                for i in range(len(bboxes)):
                    iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                    if iou_max < iou:
                        bbox = bboxes[i]
                        iou_max = iou
                landmark = self.reorder_landmark(landmark)
                if self.phase == 'train':
                    if np.random.rand() < 0.5:
                        img, _, landmark, bbox = self.hflip(
                            img, None, landmark, bbox)
                # ## 由此得到裁剪后的img landmark bbox
                img, landmark, bbox, __ = crop_face(
                    img, landmark, bbox, margin=True, crop_by_bbox=False)
                img_f, mask = self.self_blending(img.copy(), landmark.copy())
                if self.phase == 'train':
                    transformed = self.transforms(image=img_f.astype(
                        'uint8'), image1=img_f.astype('uint8'))
                    img_f = transformed['image']
                   
                img_f, landmark_last, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                    img_f, landmark, bbox, margin=False, crop_by_bbox=True, abs_coord=True, phase=self.phase)
                mask = mask[y0_new:y1_new, x0_new:x1_new]
                
                img_f = cv2.resize(
                    img_f, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
              
                img_f = img_f.transpose((2, 0, 1))

                # # 基于SSIM的一致性Map生成
                map_shape = 14  # 224/16 = 14
                
                mask_f = cv2.resize(mask, (map_shape, map_shape), interpolation=cv2.INTER_AREA).astype('float32')
                '''从v4-1之后的版本,都是直接从整脸Mask经过AREA缩放后再计算得到的xray'''
                # #这里是x ray的相关性
                # 非二值化
                mask_x_ray_f = 4 * mask_f * (1 - mask_f)
                # 二值化
                # mask_x_ray_f = np.round(4 * mask_f * (1 - mask_f))
                # mask_f = np.round(mask_f)
                #
                mask_f = self.Consistency2D(mask_f)  # ssim_patch，mask_f
                mask_x_ray_f = self.Consistency2D(mask_x_ray_f)
                flag = False
            except Exception as e:
                print(e)
                # print(idx)
                # traceback.print_exc()
                idx = torch.randint(low=0, high=len(self), size=(1,)).item()

        return img_f,  mask_f, mask_x_ray_f

    def Consistency2D(self, mask):
        real_mask = mask.reshape(1,-1)
        consis_map = [np.squeeze(1 - abs(real_mask[0,i] - real_mask))
                      for i in range(real_mask.shape[1])]
        return np.array(consis_map)

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

    def get_transforms(self):
        return alb.Compose([

            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(
                hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),

        ],
            additional_targets={f'image1': 'image'},
            p=1.)

    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1/0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def randaffine_haar(self, img, mask):
        f = alb.Affine(
            # ##haar变换里有平移会出现栅格状
            # translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1/0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )
        transformed = f(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask
    
    # ## 核心混合代码
    def self_blending(self, img, landmark):
        # H, W = len(img), len(img[0])

        if np.random.rand() < 0.25:
            landmark = landmark[:68]
            
        # 执行标准SBI
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)
        mask = get_blend_mask(mask)
        img = img.astype(np.uint8)

        return img,  mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W-landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W-bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W-bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W-bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W-bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W-bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W-bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def collate_fn(self, batch):
        img_f, mask_f,  mask_x_ray_f = zip(*batch)
        data = {}
        data['img'] = torch.tensor(img_f).float()
        data['label'] = torch.tensor([1]*len(img_f))
        data['map'] = torch.tensor(mask_f).float()
        data['map_x_ray'] = torch.tensor(mask_x_ray_f).float()
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        # # worker_seed = torch.initial_seed() % 2**32
        # # np.random.seed(worker_seed)

def convert_consis(map):
    assert map.shape==(196,196)
    consis_img = np.zeros((196,196))
    for i in range(196):
        start_h = 14*(i // 14)
        start_w = 14*(i % 14)
        for j in range(196):
            h = j // 14
            w = j % 14
            consis_img[start_h+h,start_w+w] = map[i,j]
    return consis_img


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
    import blend as B
    from PIL import Image
    from initialize import *
    from funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    from tqdm import tqdm
    if exist_bi:
        from library.vit_gen_pcl import random_get_hull, BIOnlineGeneration
        from library.oragn_mask import get_five_key, mask_patch
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # image_dataset = SBI_Dataset(phase='test', image_size=256)
    # batch_size = 64
    image_dataset = FF_Dataset(phase='train', image_size=224, n_frames=2)
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(image_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=image_dataset.collate_fn,
                                             num_workers=0,
                                             worker_init_fn=image_dataset.worker_init_fn
                                             )
    data_iter = iter(dataloader)
    # next(data_iter)
    # next(data_iter)
    data = next(data_iter)
    # print(data.keys())
    # ### DEBUG
    # for i in tqdm(range(300)):
    #         data = next(data_iter)
    # ###
    print(data['label'])
    # print(data['mask'].shape)
    img = data['img']
    map = data['map_x_ray']
    # print(img.keys())
    img = img.view((-1, 3, 224, 224))
    map = map.view((-1, 1, 196, 196))
    # img_r = img[0, :, :, :]
    # img_f = img[1, :, :, :]
    utils.save_image(img, 'imgs/loader.png', nrow=batch_size,
                     normalize=False, range=(0, 1))
    # utils.save_image(img_r, 'debug/imgs/loader_real.png', nrow=batch_size,
    #                  normalize=False, range=(0, 1))
    # utils.save_image(img_f, 'debug/imgs/loader_fake.png', nrow=batch_size,
    #                  normalize=False, range=(0, 1))
    map_f = map[1, :, :]
    utils.save_image(map, 'imgs/map.png', nrow=batch_size,
                     normalize=False, range=(0, 1))
    # utils.save_image(map_f, 'imgs/map_fake.png', nrow=batch_size,
    #                  normalize=False, range=(0, 1))
    map_f_cpu = convert_consis(torch.squeeze(map_f).cpu().data.numpy())
    Image.fromarray(np.uint8(map_f_cpu*255)).save('imgs/consis_img.png')
    if False:
        mask_0 = data['mask']
        for im in range(2):
            row = 0
            col = 0
            test_mask = mask_0[im]
            for i, mask_h in enumerate(test_mask):
                for j, mask_w in enumerate(mask_h):
                    mask_img = mask_w.detach().cpu().numpy()
                    mask_img = Image.fromarray(np.uint8(mask_img * 255), 'L')
                    mask_img.save(
                        'imgs/PCL_16/4D_mask_{}_{}_{}.png'.format(im, row, col))
                    col += 1
                row += 1
                col = 0
        print("saved")
    if False:
        import matplotlib.pyplot as plt
        for i in range(20):
            data = next(data_iter)
            # print(data.keys())
            # print(data['label'])
            img = data['img']
            # print(img.keys())
            img = img.view((-1, 3, 256, 256))
            img_r = img[0, :, :, :]
            img_f = img[1, :, :, :]
            # utils.save_image(img, 'imgs/loader_haar_rand.png', nrow=batch_size,
            #                  normalize=False, range=(0, 1))
            utils.save_image(img_r, 'debug/imgs/loader_real.png', nrow=batch_size,
                             normalize=False, range=(0, 1))
            utils.save_image(img_f, 'debug/imgs/loader_fake.png', nrow=batch_size,
                             normalize=False, range=(0, 1))

            img = cv2.imread('debug/imgs/loader_real.png')
            # img = cv2.imread('imgs/original_sequences_000_0.png')
            # img = cv2.imread(r'H:\Academic\ustc_face_forgery\Dataset\FFIW\test_set_0104/source_val_00000039_0.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img0 = np.array(img).reshape(img.shape[0], img.shape[1], 3)
            img = cv2.imread('debug/imgs/loader_fake.png')
            # img = cv2.imread('imgs/NeuralTextures_000_003_0.png')
            # img = cv2.imread(r'H:\Academic\ustc_face_forgery\Dataset\FFIW\test_set_0104/target_val_00000039_0.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = np.array(img).reshape(img.shape[0], img.shape[1], 3)
            filter1 = [[0, 0, 0, 0, 0],
                       [0, -1, 2, -1, 0],
                       [0, 2, -4, 2, 0],
                       [0, -1, 2, -1, 0],
                       [0, 0, 0, 0, 0]]
            filter2 = [[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]]
            filter3 = [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 1, -2, 1, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
            filter1 = np.asarray(filter1, dtype=float) / 4.
            filter2 = np.asarray(filter2, dtype=float) / 12.
            filter3 = np.asarray(filter3, dtype=float) / 2.
            # statck the filters
            filters = [[filter1],  # , filter1, filter1],
                       [filter2],  # , filter2, filter2],
                       [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
            filters = np.array(filters)
            # print(filters.shape)
            filters = np.repeat(filters, 3, axis=1)
            filters = torch.FloatTensor(filters)    # (3,3,5,5)
            truc = nn.Hardtanh(-3, 3)
            kernel = nn.Parameter(data=filters, requires_grad=False)
            # conv = F.conv2d(kernel, stride=1, padding=2)
            k = 0
            plt.figure(figsize=(10, 5))
            for img in [img0, img2]:

                img_show = img.copy()
                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :]
                inp = torch.Tensor(img)
                out0 = truc(F.conv2d(inp, kernel, stride=1,
                            padding=2)).detach().numpy()
                out1 = np.transpose(out0, (0, 2, 3, 1))[:, :, :, 0]
                out2 = np.transpose(out0, (0, 2, 3, 1))[:, :, :, 1]
                out3 = np.transpose(out0, (0, 2, 3, 1))[:, :, :, 2]
                # 显示
                plt.subplot(2, 4, 1 + 4*k)
                plt.xticks([]), plt.yticks([])  # 去除坐标轴
                plt.imshow(img_show.squeeze())
                plt.subplot(2, 4, 2 + 4*k)
                plt.xticks([]), plt.yticks([])  # 去除坐标轴
                plt.imshow(out1.squeeze(), cmap='bwr')
                plt.subplot(2, 4, 3 + 4*k)
                plt.xticks([]), plt.yticks([])  # 去除坐标轴
                plt.imshow(out2.squeeze(), cmap='bwr')
                plt.subplot(2, 4, 4+4*k)
                plt.xticks([]), plt.yticks([])  # 去除坐标轴
                plt.imshow(out3.squeeze(), cmap='bwr')
                k += 1
                # plt.savefig("imgs/wavelet_{}.png".format(k))
            plt.tight_layout()
            plt.savefig("debug/wavelet/BI_{}.png".format(i))
else:
    from utils import blend as B
    from .initialize import *
    from .funcs import IoUfrom2bboxes, crop_face, RandomDownScale
    if exist_bi:
        from utils.library.vit_gen_pcl import random_get_hull, BIOnlineGeneration
        from utils.library.oragn_mask import get_five_key, mask_patch
