from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class RocksDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transform_image, transform_mask) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.masks_dir = Path(masks_dir)
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.images = [img for img in self.image_dir.glob('**/*') if img.is_file()]
        self.masks = [mask for mask in self.masks_dir.glob('**/*') if mask.is_file()]

    def __getitem__(self, index):
        img_path = Image.open(self.images[index]).convert("RGB")
        mask_path = Image.open(self.masks[index]).convert("L")
        image = self.transform_image(img_path)
        mask = self.transform_mask(mask_path)
        return image, mask
    
    def __len__(self):
        return len(self.images)