from pydub import AudioSegment, silence
from moviepy.editor import *
from pathlib import Path
from os import path


def split_videos(filename, output):
    global silence
    # Split file on silences
    video = AudioSegment.from_file(filename, "mp4")
    silences = silence.detect_nonsilent(video, silence_thresh=-40, min_silence_len=400)

    print("Done splitting audio, start splitting clips")

   
def main():
    folder = Path('data')
    files = [x for x in folder.iterdir()]
    os.makedirs('./output', exist_ok=True)
    for i, filename in enumerate(files):
        filename = str(filename)
        output = "./output/data/{}".format(i)
        print(output,"\n", filename)
        os.makedirs(output, exist_ok=True)
        split_videos(filename, output)


if __name__ == "__main__":
    main()
