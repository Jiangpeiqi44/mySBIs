
import torch
import numpy as np
import random

# from utils.ibi_wavelet import SBI_Dataset
# from utils.bi_wavelet import SBI_Dataset
# from utils.sbi_default import SBI_Dataset
from utils.DEBUG_dataset import SBI_Dataset

from utils.funcs import load_json
from tqdm import tqdm

from torchvision import  utils

def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
        
def main():
    seed = 3   # 默认 seed = 5
    seed_torch(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False  # False

    image_size = 224 #cfg['image_size']
    batch_size = 2 #cfg['batch_size']
    train_dataset = SBI_Dataset(
        phase='train', image_size=image_size, n_frames=1)
    val_dataset = SBI_Dataset(phase='val', image_size=image_size, n_frames=1)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size//2,
                                               shuffle=False,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=1,
                                               pin_memory=False,
                                               drop_last=True,
                                            #    worker_init_fn=train_dataset.worker_init_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=val_dataset.collate_fn,
                                             num_workers=1,
                                             pin_memory=False,
                                             worker_init_fn=val_dataset.worker_init_fn
                                             )

    n_epoch = 7
    for epoch in range(n_epoch):
        # seed_torch(seed+epoch)
        print("epoch {}".format(epoch))
        np.random.seed(seed+epoch)
        # random.seed(seed)
        # np.random.seed(seed)
        epoch_first_data = True
        for step, data in enumerate((val_loader)):
            img = data['img']
            if epoch_first_data:
                epoch_first_data = False
                img = img.view((-1, 3, 224, 224))
                utils.save_image(img, 'imgs/loader_{}.png'.format(epoch), nrow=batch_size,
                     normalize=False, range=(0, 1))
            else:
                break


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument(dest='config')
    # parser.add_argument('-n', dest='session_name')
    # parser.add_argument('-w', dest='weight_name')
    # args = parser.parse_args()
    main()
