from PIL import Image
from pathlib import Path

def augment(folder):
    files = [img for img in folder.glob('**/*') if img.is_file()]
    for file in files:
        name = file.parts[-1].split('.')
        img = Image.open(file)
        rot_img = [img.rotate(45*i) for i in range(1,8)]
        rot_names = [folder/(f"{name[0]}_rotate_{45*i}.{name[1]}") for i in range(len(rot_img))]
        for i in range(len(rot_img)):
            rot_img[i].save(rot_names[i])

        flip_img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        flip_img.save(folder/f"{name[0]}_flip_left_right.{name[1]}")
        flip_img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        flip_img.save(folder/f"{name[0]}_flip_top_bottom.{name[1]}")

if __name__ == "__main__":
    img_folder = Path("data/converted_img")
    mask_folder = Path("data/mask")
    augment(img_folder)
    augment(mask_folder)