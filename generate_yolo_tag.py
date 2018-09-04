from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import tensorflow as tf
from utils import label_map_util
import cv2
import numpy as np

input_label_map = '/home/chamo/Documents/work/OpenImgChamo/config/oid_object_detection_challenge_500_label_map.pbtxt'
input_box_annotations_csv = '/home/chamo/Documents/data/openImg/train/box.csv.chamo'
input_images_directory = '/media/chamo/c769deb7-4cd0-4af7-a517-c5601d8313ad/train'
ouput_file='/home/chamo/Documents/work/keras-yolo3/all.txt'
ouput_class_name='/home/chamo/Documents/work/keras-yolo3/model_data/open_img_code.txt'

keyword=['Tripod']

def get_class_dict(label_map):
    categories = []
    for item in label_map.item:
        categories.append(item.name+'\n')
    return categories

def main2(_):
    label_map = label_map_util.load_labelmap(input_label_map)
    categories = get_class_dict(label_map)
    with open(ouput_class_name, "w") as f_out:
        f_out.writelines(categories)

def main1(_):
    #ff4f0fbb880080b7
    count=0
    with open(input_box_annotations_csv+'.chamo1', "w") as f_output:
        with open(input_box_annotations_csv, "r") as f_input:
            line = f_input.readline()
            last_img_name=''
            last_img_shape=None
            while True:
                count=count+1
                line = f_input.readline()
                if line == '':
                    break
                splited_str = line.split(",")
                if last_img_name==splited_str[0]:
                    img_size=last_img_shape
                else:
                    image_path = os.path.join(input_images_directory, splited_str[0] + '.jpg')
                    img=cv2.imread(image_path)
                    if not hasattr(img,'shape'):
                        print(splited_str[0])
                        continue
                    img_size = img.shape
                    last_img_name=splited_str[0]
                    last_img_shape=img_size
                splited_str[4]=str(int(float(splited_str[4])*img_size[1]))
                splited_str[5] = str(int(float(splited_str[5]) * img_size[1]))
                splited_str[6] = str(int(float(splited_str[6]) * img_size[0]))
                splited_str[7] = str(int(float(splited_str[7]) * img_size[0]))
                line=','.join(splited_str)
                f_output.write(line)
                if count % 1000 == 0:
                    print(count)

def main(_):
    label_map = label_map_util.get_label_map_dict(input_label_map)
    all_box_annotations = pd.read_csv(input_box_annotations_csv)
    all_images = tf.gfile.Glob(os.path.join(input_images_directory, '*.jpg'))
    all_image_ids = [os.path.splitext(os.path.basename(v))[0] for v in all_images]
    all_image_ids = pd.DataFrame({'ImageID': all_image_ids})
    all_annotations = pd.concat([all_box_annotations, all_image_ids])
    count=0
    class_count_list=[]
    for i in range(500):
        class_count_list.append(0)
    with open(ouput_file, "w") as f_out:
        for counter, image_data in enumerate(all_annotations.groupby('ImageID')):

            image_id, image_annotations = image_data
            filtered_data_frame = image_annotations[image_annotations.LabelName.isin(label_map)]
            if len(filtered_data_frame) == 0:
                continue
            filtered_data_frame_boxes = filtered_data_frame[~filtered_data_frame.YMin.isnull()]
            class_id_np = filtered_data_frame_boxes.LabelName.map(lambda x: label_map[x]).as_matrix()
            is_ok=False
            for item in class_id_np:
                class_count_list[item - 1] = class_count_list[item - 1] + 1
                if class_count_list[item - 1] < 500:
                    is_ok=True
                    break
            if not is_ok:
                continue
            image_path = os.path.join(input_images_directory, image_id + '.jpg')
            #img_size=cv2.imread(image_path).shape

            ymin_np = filtered_data_frame_boxes.YMin.as_matrix()
            xmin_np = filtered_data_frame_boxes.XMin.as_matrix()
            ymax_np = filtered_data_frame_boxes.YMax.as_matrix()
            xmax_np = filtered_data_frame_boxes.XMax.as_matrix()
            if len(ymin_np)<1:
                continue
            count = count + 1
            boxes_str=''
            for i in range(len(ymin_np)):
                box_str=str(int(xmin_np[i]))+','+str(int(ymin_np[i]))+','+str(int(xmax_np[i]))+','+str(int(ymax_np[i]))+','+str(int(class_id_np[i]-1))
                boxes_str=boxes_str+' '+box_str
            if box_str=='':
                continue
            line_str=image_path+boxes_str+'\n'
            f_out.write(line_str)
            if count%1000==0:
                print(count)

if __name__ == '__main__':
    tf.app.run()
