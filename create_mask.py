import json
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import draw

MASK_PATH = "data/mask/"
JSON_FILE_PATH = "data/data.json"
IMG_DIR = "data/converted_img/"
IMG_SIZE = 128

mask_folder = Path(MASK_PATH)
mask_folder.mkdir(parents=True, exist_ok=True)

coordinates = json.load(open(JSON_FILE_PATH))

for image_num in range(len(coordinates)):
    img_name = coordinates[image_num]['image'].split('-')[1]
    mask = np.zeros((IMG_SIZE,IMG_SIZE), dtype=bool)
    if 'label' in coordinates[image_num]:
        for label in range(len(coordinates[image_num]['label'])):
            temp_mask = coordinates[image_num]['label'][label]['points']
            for a in temp_mask:
                for i in range(2):
                    a[i] = a[i] / 100*IMG_SIZE
            temp_mask = draw.polygon2mask((IMG_SIZE,IMG_SIZE), temp_mask)
            mask = np.add(mask, temp_mask)
        mask = np.rot90(mask, 3)
        mask = np.fliplr(mask)
    
    savemask = np.array(mask, dtype=np.uint8)*255
    i = Image.fromarray(savemask, mode='L')
    i.save(MASK_PATH + img_name)