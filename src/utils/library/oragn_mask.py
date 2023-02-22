# add
import math
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
import numpy as np

def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
    reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [tuple(x.astype('int32')) for x in [
        leye_center, reye_center, nose, lmouth, rmouth, leye_left, leye_right, reye_left, reye_right
    ]]
    return out


def remove_eyes(image, landmarks, opt):
    # l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line


def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def random_deform(mask, nrows, ncols, mean=0, std=10):
    h, w = mask.shape[:2]
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:, 0], 0, h-1, deformed[:, 0])
    np.clip(deformed[:, 1], 0, w-1, deformed[:, 1])
    trans = PiecewiseAffineTransform()
    trans.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, trans)
    warped *= mask
    mask_bi = warped.copy()
    blured = cv2.GaussianBlur(warped, (5, 5), 3)  # sigma = 3
    return blured, mask_bi


def mask_patch(reg, img, five_key):
    if reg == 0:
        mask_patch = remove_eyes(img, five_key, 'l')
    elif reg == 1:
        mask_patch = remove_eyes(img, five_key, 'r')
    elif reg == 2:
        mask_patch = remove_eyes(img, five_key, 'b')
    elif reg == 3:
        mask_patch = remove_nose(img, five_key)
    elif reg == 4:
        mask_patch = remove_mouth(img, five_key)
    elif reg == 5:
        mask_patch = remove_nose(
            img, five_key) + remove_eyes(img, five_key, 'l')
    elif reg == 6:
        mask_patch = remove_nose(
            img, five_key) + remove_eyes(img, five_key, 'r')
    elif reg == 7:
        mask_patch = remove_nose(
            img, five_key) + remove_eyes(img, five_key, 'b')
    elif reg == 8:
        mask_patch = remove_nose(
            img, five_key) + remove_mouth(img, five_key)
    elif reg == 9:
        mask_patch = remove_eyes(
            img, five_key, 'b') + remove_nose(img, five_key) + remove_mouth(img, five_key)
    mask_patch, mask_bi = random_deform(
        (mask_patch*255).astype(np.uint8), 5, 5)
    return (mask_patch/255).reshape(mask_patch.shape+(1,)), (mask_bi/255).reshape(mask_bi.shape+(1,))
# add done
