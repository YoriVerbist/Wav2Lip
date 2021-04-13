from pathlib import Path
import os
import subprocess
import re


# Crop left half of video
template = 'ffmpeg -i {} -filter:v "crop=in_w/2:in_h:in_w/2:0" {}'
folder  = Path('output/data')
dirs = [x for x in folder.iterdir()]
os.makedirs('output/cropped', exist_ok=True)
for i, dir_name in enumerate(dirs):
    files = [x for x in dir_name.iterdir()]
    dir_name = str(dir_name)
    os.makedirs('./output/cropped/{}'.format(dir_name[12:]), exist_ok=True)
    for file_name in files:
        file_name = str(file_name)
        destination = "output/cropped/{}/{}".format(dir_name[12:], file_name[file_name.find("000"):])
        command = template.format(file_name, destination)
        subprocess.call(command, shell=True)
