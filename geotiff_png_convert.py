from pathlib import Path
from osgeo import gdal

gdal.SetConfigOption("GDAL_PAM_ENABLED", "FALSE") # delete if geo info is needed

conv_img = "data/converted_img/"
converted_folder = Path(conv_img)
converted_folder.mkdir(parents=True, exist_ok=True)

tif_folder = Path("data/tiff_img").glob('**/*')
file_list = [x for x in tif_folder if x.is_file()]

options_list = [
    '-ot Byte',
    '-of PNG',
    '-b 1',
    '-b 2',
    '-b 3',
    '-scale'
]

options_string = " ".join(options_list)

for img in file_list:
    gdal.Translate(str(converted_folder/img.stem)+'.png',str(img),options=options_string)

