import torch.utils.data as data

from data import common


class LRDataset(data.Dataset):
    '''
    Read LR images only in test phase.
    '''

    def name(self):
        return common.find_benchmark('/64res')      # directory of LR imgs


    def __init__(self, path):
        super(LRDataset, self).__init__()
        self.oath = path
        self.scale = 2      # scale of task (2x)
        self.paths_LR = None

        # read image list from image/binary files
        self.paths_LR = common.get_image_paths("img", path)
        assert self.paths_LR, '[Error] LR paths are empty.'


    def __getitem__(self, idx):
        # get LR image
        lr, lr_path = self._load_file(idx)
        lr_tensor = common.np2Tensor([lr], 255)[0]
        return {'LR': lr_tensor, 'LR_path': lr_path}


    def __len__(self):
        return len(self.paths_LR)


    def _load_file(self, idx):
        lr_path = self.paths_LR[idx]
        lr = common.read_img(lr_path, "img")

        return lr, lr_path
