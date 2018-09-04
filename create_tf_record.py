import numpy as np
import os
import tensorflow as tf
import io
import logging
import random
import sys
import PIL.Image
import hashlib

sys.path.append("/home/chamo/Documents/work/output/models/research/")
from utils import dataset_util
from utils import label_map_util


def get_examples(img_path):
    
    label_path =os.path.splitext(img_path)[0]+'.txt'
    if os.path.exists(label_path) is False:
        return False,None
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        print("file format error "+img_path)
        return False,None
    key = hashlib.sha256(encoded_jpg).hexdigest()    
    examples=[]
    for line in open(label_path):  
        data=line.split( )
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []

        
        width=data[0]
        height=data[1]
        file_name=data[2]
        image_format=data[3]
        xmin.append(float(data[4]))
        xmax.append(float(data[5]))
        ymin.append(float(data[6]))
        ymax.append(float(data[7]))
        classes.append(int(data[8]))
        classes_text.append(data[9].encode('utf8'))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(int(height)),
          'image/width': dataset_util.int64_feature(int(width)),
          'image/filename': dataset_util.bytes_feature(file_name.encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(file_name.encode('utf8')),
          'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        examples.append(example)

    return True,examples    

def create_tf_record(examples_list,output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for i in examples_list:
        ret,examples=get_examples(i)
        if ret:
            for j in examples:
                writer.write(j.SerializeToString())
    writer.close()   


def main(image_dir='./image/',tfrecord_dir='./',train_percent=0.9):
    images=[]
    for (root,dirs,files) in os.walk(image_dir) :
        for item in files:
            if item.endswith('jpg'):
                images.append(os.path.join(root, item))
    
    random.seed(42)
    random.shuffle(images)
    num_examples = len(images)
    num_train = int(train_percent* num_examples)
    train_examples = images[:num_train]
    val_examples = images[num_train:]
    
    train_output_path = os.path.join(tfrecord_dir, 'train.record')
    val_output_path = os.path.join(tfrecord_dir, 'val.record')
    
    create_tf_record(train_examples,train_output_path)
    create_tf_record(val_examples,val_output_path)
    print('complete task!\n crate %d training at %s,\n and %d validation examples at %s.'%(len(train_examples),train_output_path, len(val_examples),val_output_path))
    
    
main()
