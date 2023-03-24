from torch.utils.data import DataLoader
from dataset import RocksDataset
import torch

def load_data_set(image_dir, masks_dir, transforms, batch_size=32, shuffle=True):
    dataset = RocksDataset(image_dir,
                           masks_dir,
                           transform_image=transforms[0],
                           transform_mask=transforms[1])
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.7, 0.3])

    return DataLoader(train_dataset,
                      batch_size=batch_size,
                      shuffle=shuffle
                      ), DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle
                                    )