import os
import glob
from tqdm import tqdm
import numpy as np
import json
from skimage import io
import cv2



def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

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

def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark
dataset_path = 'H:/Academic/ustc_face_forgery/Dataset/FF++/original_sequences/youtube/raw/landmarks/'
lm_all_path = glob.glob(dataset_path+'*/*.npy')
lm_all_path = [i.replace('\\','/') for i in lm_all_path]
# print(lm_all_path[:10])
## 筛选train val test
filelist = []
list_dict = json.load(
    open(f'H:/Academic/ustc_face_forgery/Dataset/FF++/train.json', 'r'))
# print(list_dict)
for i in list_dict:
    filelist += i
# print(filelist)
lm_all_path = [i for i in lm_all_path if i.split('/')[-2] in filelist]
# print(len(lm_all_path))
# ###
print(len(lm_all_path))
err_dict = {}
lm_dict = {}
for lm_path in tqdm(lm_all_path):
        vid_idx = lm_path.split('/')[-2]
        frame_idx = lm_path.split('/')[-1].split('.')[0]
        lm_idx = '{}_{}.png'.format(vid_idx,frame_idx)
        frame_path = lm_path.replace('/landmarks/','/frames/').replace('npy','png')
        img = io.imread(frame_path)
        landmark = np.load(lm_path)[0]
        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(
        ), landmark[:, 0].max(), landmark[:, 1].max()])
        bboxes = np.load(frame_path.replace(
            '.png', '.npy').replace('/frames/', '/retina/'))[:2]
        iou_max = -1
        for i in range(len(bboxes)):
            iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
            if iou_max < iou:
                bbox = bboxes[i]
                iou_max = iou
        landmark = reorder_landmark(landmark)
        img, landmark, bbox, ___ = crop_face(
            img, landmark,bbox, margin=True, crop_by_bbox=False)
        img, landmark_last, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(
                        img, landmark, bbox, margin=False, crop_by_bbox=True, abs_coord=True, phase='train')
        try:
            # TODO
            img = cv2.resize(
                    img, (224,224), interpolation=cv2.INTER_LINEAR).astype('float32')/255
            err_dict[lm_idx] = True
            lm_dict[lm_idx] = landmark.tolist()
        except Exception as e:
            print(e)
            err_dict[lm_idx] = False
            print('set '+lm_idx+' to False')
dict_json=json.dumps(err_dict)#转化为json格式文件
dict_json_lm=json.dumps(lm_dict)#
# 将json文件保存为.json格式文件
with open('src/utils/err_face_train.json','w+') as file:
    file.write(dict_json)
with open('src/utils/library/ff_lm_train_clean.json','w+') as file:
    file.write(dict_json_lm)
# print(lm_all_path[:10])

