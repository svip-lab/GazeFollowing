import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str,
                    help='path to clip video dir',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted frames',
                    required=True)
args = parser.parse_args()

video_dir = args.video_dir
output_dir = args.output_dir

for file_name in glob.glob(os.path.join(video_dir, '*/*')):
    if not (file_name.endswith(".mp4") or file_name.endswith(".mov")):
        continue
    video = file_name.split('/')[-1]
    path = os.path.join(output_dir, video[:-4])

    if not os.path.exists(path):
        os.makedirs(path)
    os.system('ffmpeg -i '+ '\"'+ file_name+'\"' + ' -r 2 ' + '\"' +path + '/%05d.png' + '\"')


