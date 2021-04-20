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
    
    print(filename)
    clip = VideoFileClip(filename)
    for i, silence in enumerate(silences):
        start = silence[0]/1000
        end = silence[1]/1000
        length = end - start
        long_vid = False
        while length > 10:
            long_vid = True
            temp = start + 5
            subpart = clip.subclip(start, temp)
            subpart.write_videofile("{}/000s{}.mp4".format(output, i))
            i += 1
            start += 5
            length -= 5
        if long_vid or length < 1:
            continue
        subpart = clip.subclip(start, end)
        subpart.write_videofile("{}/000{}.mp4".format(output, i))

   
def main():
    folder = Path('../data')
    os.makedirs('../output', exist_ok=True)
    for i, filename in enumerate(folder.iterdir()):
        filename = str(filename)
        output = "../output/data/{}".format(i)
        print(output,"\n", filename)
        os.makedirs(output, exist_ok=True)
        split_videos(filename, output)


if __name__ == "__main__":
    main()
