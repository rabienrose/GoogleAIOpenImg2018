import tensorflow as tf
import numpy as np
from utils import label_map_util
import cv2
import random
tfrecords_filename = "/media/chamo/e9cbf274-e538-4ccc-adbb-16cc0932f014/train_tfrecord/train-00000-of-00200"
class_name_file = './config/oid_object_detection_challenge_500_label_map.pbtxt'

def get_class_dict(label_map):
    categories = {}
    for item in label_map.item:
        categories[item.name]=item.display_name
    return categories

label_map = label_map_util.load_labelmap(class_name_file)
max_num_classes = max([item.id for item in label_map.item])
categories = get_class_dict(label_map)
 
filename_queue = tf.train.string_input_producer([tfrecords_filename]) 
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue) 
     
features = tf.parse_single_example(serialized_example,
                                   features={
                                        'image/filename':  tf.FixedLenFeature([], tf.string),
                                        'image/source_id': tf.FixedLenFeature([], tf.string),
                                        'image/encoded': tf.FixedLenFeature([], tf.string),
                                        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                                        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                                        'image/object/bbox/ymin':tf.VarLenFeature(tf.float32),
                                        'image/object/bbox/ymax':tf.VarLenFeature(tf.float32),
                                        'image/object/class/text':tf.VarLenFeature(tf.string),
                                        'image/object/class/label':tf.VarLenFeature(tf.int64),
                                   })
filename = tf.cast(features['image/filename'], tf.string)
xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
text = tf.cast(features['image/object/class/text'], tf.string)
label = tf.cast(features['image/object/class/label'], tf.int64)
image =tf.image.decode_jpeg(features['image/encoded'])

with tf.Session() as sess: 
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        image1,xmin1s,xmax1s,ymin1s,ymax1s, class_id1, label1=sess.run([image,xmin,xmax,ymin,ymax, text, label])
        height1,width1,_=image1.shape
        box_count=len(xmin1s[1])
        class_color_dict = {}
        for j in range(box_count):
            if class_id1[1][j] not in class_color_dict.keys():
                class_id_str=class_id1[1][j].decode()
                class_color_dict[class_id_str] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            xmin1 = xmin1s[1][j]
            ymin1 = ymin1s[1][j]
            xmax1 = xmax1s[1][j]
            ymax1 = ymax1s[1][j]
            xmin1,ymin1=int(xmin1*width1),int(ymin1*height1)
            xmax1,ymax1=int(xmax1*width1),int(ymax1*height1)
            cv2.rectangle(image1, (xmin1, ymin1), (xmax1, ymax1), class_color_dict[class_id_str], 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            x_sift = int(20 * len(categories[class_id_str]) / 2)
            image1 = cv2.putText(image1, categories[class_id_str], (int((xmin1 + xmax1) / 2) - x_sift, int((ymin1 + ymax1) / 2) + 10),
                              font, 1, class_color_dict[class_id_str], 2)
        cv2.imshow("Image", image1)
        cv2.waitKey(0)
    coord.request_stop()
    coord.join(threads)
