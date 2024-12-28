import os
from torch.utils import data
from dataset.dataset import RainDropDataset

def get_loader(basename):
    
    return RainDropDataset

def get_loader_train(config):
    if config.train_dir is None:
        return None
    basename = os.path.basename(config.train_dir)
    data_reader = get_loader(basename)
    
    # Ensure that img_size is included in the config
    dataset = data_reader(config.train_dir, img_size=config.img_size, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,  # Consider setting shuffle to True during training
                                  num_workers=config.num_workers)
    return data_loader

def get_loader_val(config):
    if config.val_dir is None:
        return None
    basename = os.path.basename(config.val_dir)
    data_reader = get_loader(basename)
    
    # Ensure that img_size is included in the config
    dataset = data_reader(config.val_dir, img_size=config.img_size, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader

def get_loader_test(config):
    if config.test_dir is None:
        return None
    basename = os.path.basename(config.test_dir)
    data_reader = get_loader(basename)
    
    # Ensure that img_size is included in the config
    dataset = data_reader(config.test_dir, img_size=config.img_size, length=config.length)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    return data_loader
