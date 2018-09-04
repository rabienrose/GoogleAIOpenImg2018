import cv2
from utils import label_map_util
import random

box_file = '/home/chamo/Documents/data/UntitledFolder/test.csv'
img_dir = '/home/chamo/Documents/data/UntitledFolder/test_challenge_2018'
#img_dir = '/media/chamo/e9cbf274-e538-4ccc-adbb-16cc0932f014/validation'
#img_dir = '/home/chamo/Documents/data/human'
class_name_file = '/home/chamo/Documents/work/models/research/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt'


def get_class_dict(label_map):
    categories = {}
    for item in label_map.item:
        categories[item.name]=item.display_name
    return categories


label_map = label_map_util.load_labelmap(class_name_file)
max_num_classes = max([item.id for item in label_map.item])
categories = get_class_dict(label_map)
with open(box_file, "r") as f:
    line = f.readline()
    while line:
        line = f.readline()
        splited = line.split(",")
        image_name = splited[0]
        img = cv2.imread(img_dir + '/' + image_name + '.jpg')
        print(img_dir + '/' + image_name + '.jpg')
        width=img.shape[1]
        height = img.shape[0]
        class_color_dict = {}
        if len(splited) > 1:
            box_list_str = splited[1]
            splited = box_list_str.split(" ")
            if len(splited) > 1:
                box_count = 0
                while True:
                    class_id = splited[box_count* 6 + 0]
                    if class_id not in class_color_dict.keys():
                        class_color_dict[class_id]=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    xmin = float(splited[box_count* 6 + 2])
                    ymin = float(splited[box_count* 6 + 3])
                    xmax = float(splited[box_count* 6 + 4])
                    ymax = float(splited[box_count* 6 + 5])
                    xmin = int(xmin * width)
                    xmax = int(xmax * width)
                    ymin = int(ymin * height)
                    ymax = int(ymax * height)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), class_color_dict[class_id], 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x_sift=int(20*len(categories[class_id])/2)
                    img = cv2.putText(img, categories[class_id], (int((xmin+xmax)/2)-x_sift, int((ymin+ymax)/2)+10), font, 1, class_color_dict[class_id], 2)
                    box_count = box_count + 1
                    if len(splited)-1 <= box_count * 6:
                        break
        cv2.imshow("Image", img)
        cv2.waitKey(0)
