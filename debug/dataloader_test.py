import torch
from torch.utils.data import Dataset
import numpy as np
import random

class TestDataset(Dataset):
    def __init__(self):
        self.datas = np.arange(16)
        print('init')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        random_data = np.random.uniform(0.0, 1.0)
        return  data, random_data
    
    def worker_init_fn(self, worker_id):
        # np.random.seed(np.random.get_state()[1][0] + worker_id)
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

if __name__ == '__main__':
    simple_dataset = TestDataset()
    dataloader = torch.utils.data.DataLoader(simple_dataset, 
                                             batch_size=2,
                                             shuffle=False,
                                             worker_init_fn=simple_dataset.worker_init_fn,
                                             num_workers=1)
    n_epoch = 2
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    for epoch in range(n_epoch):
        print('epoch_%d'%epoch)
        np.random.seed(seed + epoch)
        for step, data in enumerate(dataloader):
            print(data)