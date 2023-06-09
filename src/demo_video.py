import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import math
from torchvision import datasets, transforms, models, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from PIL import Image
import sys
import random
import shutil
# from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from inference.preprocess import extract_face,extract_face_max,extract_face_and_bbox
from vit_consis_model import Vit_hDRMLPv2_consisv5 as Detector
# from model_zoo import select_model as Detector
import warnings
import skimage.io as io
from skimage import transform
import glob
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, roc_curve


def main(args,face_detector,model,img):

    
    device = torch.device('cuda')
    try:
        # face_list=[(transform.resize(io.imread(img), (image_size, image_size))).transpose((2, 0, 1))]
        # frame = cv2.imread(img)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_list, x0, y0, x1, y1  = extract_face_and_bbox(img, face_detector,image_size=(224,224))
        # face_list = frame.reshape((1,)+frame.shape)
    except Exception as e:
            print(e)
            print(img)
            return
    with torch.no_grad():
        img_t = torch.tensor(face_list).to(device).float()/255
        # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
        # pred = model.test_time(img_t).softmax(1)[:, 1].cpu().data.numpy().tolist() #.test_time
        pred = model(img_t).softmax(1)[:, 1].cpu().data.numpy().tolist() #.test_time
    # if max(pred) < 0.5:
    #     print(img)
    print(f'fakeness: {max(pred):.4f}')

    return max(pred), x0, y0, x1, y1
    
if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', dest='weight_name', type=str)
    parser.add_argument('-i', dest='input_vid', type=str)
    args = parser.parse_args()
    model = Detector()
    # model = Detector('EFNB4')
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()
    face_detector = get_model("resnet50_2020-07-20",
                              max_size=2048, device=device)
    face_detector.eval()

    import os.path
    import time
    import cv2
    
    video_path = args.input_vid
    save_path = video_path[:video_path.rfind('.')]
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('FPS:{:.2f}'.format(fps))
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num/rate
    print('video total time:{:.2f}s'.format(duration))
    
    # width, height = 1920, 1080
    # interval = int(fps) * 4
    interval = 1
    process_num = frame_num // interval
    print('process frame:{:.0f}'.format(process_num))
    
    cnt = 0
    num = 0
    
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cnt += 1
            if cnt % interval == 0:
                num += 1
                # frame = cv.resize(frame, (width, height))
                # cv2.imwrite(save_path + "/%07d.jpg" % num, frame)
                remain_frame = process_num - num
                pred, x0, y0, x1, y1  = main(args,face_detector,model,img)
                print(pred)
                t1 = time.time() - t0
                t0 = time.time()
                print("Processing %07d.jpg, remain frame: %d, remain time: %.2fs" % (num, remain_frame, remain_frame * t1))
        else:
            break
        if cv2.waitKey(1) & 0xff == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    print("done")










