import torch
# from torch.utils.data import Dataset,DataLoader
# import numpy as np
# import random


def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
        
# class TestDataset(Dataset):
#     def __init__(self):
#         self.datas = np.arange(8)
#         print('init')

#     def __len__(self):
#         return len(self.datas)

#     def __getitem__(self, index):
#         data = self.datas[index]
#         print(np.random.get_state()[1][0:5])
#         random_data = [np.random.randint(100),np.random.randint(100)]
#         return  data, random_data
    
# def worker_init_fn(worker_id):
#     print('1',np.random.get_state()[1][0:5])
#     np.random.seed(np.random.get_state()[1][0] + worker_id)
#     print('2',np.random.get_state()[1][0:5])
#     # print(torch.utils.data.get_worker_info())
#     # print(torch.initial_seed())
#     # worker_seed = torch.initial_seed() % 2**32
#     # np.random.seed(worker_seed)

# if __name__ == '__main__':
#     simple_dataset = TestDataset()
#     dataloader = DataLoader(simple_dataset, 
#                             batch_size=4,
#                             shuffle=False,
#                             # worker_init_fn=worker_init_fn,
#                             num_workers=1)
#     n_epoch = 2
#     seed = 42
#     seed_torch(seed)
    # print(np.random.randint(10))
    # torch.cuda.manual_seed(seed)
    # for epoch in range(n_epoch):
    #     print('epoch_%d'%epoch)
    #     seed_torch(seed+epoch)
    #     for data in dataloader:
    #         print(data)

import numpy as np
import random

# np.random.seed(0)

class Transform(object):
    def __init__(self):
        pass

    def __call__(self, item = None):
        return [np.random.randint(10000, 20000), random.randint(20000,30000)]

class RandomDataset(object):

    def __init__(self):
        pass

    def __getitem__(self, ind):
        item = [ind, np.random.randint(1, 10000), random.randint(10000, 20000), 0]
        tsfm =Transform()(item)
        # print(np.random.rand())
        return np.array(item + tsfm)
    def __len__(self):
        return 20


from torch.utils.data import DataLoader

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

ds = RandomDataset()
ds = DataLoader(ds, 10, shuffle=False, num_workers=1,worker_init_fn=worker_init_fn) #, worker_init_fn=worker_init_fn
seed = 42
# np.random.seed(seed)
for epoch in range(2):
    print("epoch {}".format(epoch))
    # seed_torch(seed)
    np.random.seed(seed)
    for batch in ds:
        print(batch)