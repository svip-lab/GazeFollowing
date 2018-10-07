import os
import json
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--Images_dir', type=str,
                    help='path of extracted video frames',
                    required=True)
parser.add_argument('--Video_annotation_dir', type=str,
                    help='path of gaze annotations',
                    required=True)
parser.add_argument('--Output_json_file', type=str,
                    help='path to store annotation_list',
                    required=True)

args = parser.parse_args()

#images_data_root = '/ssd/yuzh/Gaze/our_data/'
#annotation_root = '/ssd/yuzh/Gaze/video_annotation/'

images_data_root = args.Images_dir
annotation_root = args.Video_annotation_dir
output_json_file = args.Output_json_file

paths = []
boxes = []
points = []

for root, dirs, files in os.walk(annotation_root):
    for file_name in files:
        if file_name.endswith('.json'):
            orig_dir = os.path.join(images_data_root, file_name[:-5])
            assert(os.path.exists(os.path.join(images_data_root, file_name[:-5])))
            images_dir = os.path.join(images_data_root, file_name[:-5])
            jsonfile = os.path.join(root, file_name)
            jsondata = json.load(open(jsonfile))

            for key in jsondata:
                if(len(jsondata[key]['regions'].keys()) <= 0):
                    continue
                regions = jsondata[key]['regions']
                ann_type_set = set([])
                ann_type_2_index = {}
                for i in regions.keys():
                    ann_type_set.add(regions[i]['shape_attributes']['name'])
                    ann_type_2_index[regions[i]['shape_attributes']['name']] = i
                if 'rect' in ann_type_set and 'point' in ann_type_set:
                    rect = ann_type_2_index['rect']
                    point = ann_type_2_index['point']
                    #print(regions[rect], regions[point])
                    filename = jsondata[key]['filename']
                    if not os.path.exists(os.path.join(images_dir, filename)):
                        #print(os.path.join(images_dir, filename))
                        break

                    paths.append(os.path.join(images_dir, filename))
                    boxes.append((regions[rect]['shape_attributes']['x'],
                                  regions[rect]['shape_attributes']['y'],
                                  regions[rect]['shape_attributes']['width'],
                                  regions[rect]['shape_attributes']['height']))
                    points.append((regions[point]['shape_attributes']['cx'],
                                   regions[point]['shape_attributes']['cy']))


 
print(len(paths), len(boxes), len(points))       
json.dump({"path": paths, "boxes": boxes, "points" : points}, open(output_json_file, 'w'))

