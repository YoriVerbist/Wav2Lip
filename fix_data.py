from pathlib import Path
from glob import glob
from os import path
import librosa

i = 0
folder = Path('filelists')
for dirs in folder.iterdir():
    for videos in dirs.iterdir():
        images = glob(path.join(videos, '*.jpg'))
        audiofile = glob(path.join(videos, '*.wav'))[0]
        duration = round(librosa.get_duration(filename=audiofile) * 25)
        #print(duration)
        if len(images) == duration:
            i += 1
            #print('remove this', videos)
print(i)
