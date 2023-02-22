import os 
import shutil
import math
base_dir = 'imagenet100_train'
target_dir = 'imagenet100_val'

classes = os.listdir(base_dir)
ratio=0.2
for class0 in classes:
    os.makedirs(os.path.join(target_dir,class0))
    pics = os.listdir(os.path.join(base_dir,class0))
    pics = pics[:math.floor(len(pics)*ratio)]
    for pic in pics:
        shutil.move(os.path.join(base_dir,class0,pic), os.path.join(target_dir,class0,pic))
        print('Moving' + pic)