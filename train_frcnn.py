
import tensorflow as tf
from core import box_list
from nets.yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
import cv2
from PIL import Image, ImageDraw
import numpy as np

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

tfrecord = '/home/chamo/Documents/data/val_tfrecord/val_tfrecord-00000-of-00100'
#tfrecord = '/media/chamo/e9cbf274-e538-4ccc-adbb-16cc0932f014/train_tfrecord/train-00000-of-00200'
output_addr ='/home/chamo/Documents/data/output'
weights_file='/home/chamo/Documents/data/yolov3.weights'
classes_file='/home/chamo/Documents/data/coco.names'
batch_size=1
class_num=500
image_size=416

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

def draw_boxes(boxes_list, imgs, cls_names, detection_size, img_names):
    img_count=0
    for boxes in boxes_list:
        img=np.uint8(imgs[img_count,:,:,:])
        img_name=img_names
        imgimg = Image.fromarray(img)
        draw = ImageDraw.Draw(imgimg)
        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))
            for box, score in bboxs:
                box = convert_to_original_size(box, np.array(detection_size), np.array(imgimg.size))
                draw.rectangle(box, outline=color)
                draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
        imgimg.save(output_addr +'/' +img_name.decode()+'.jpg')
        img_count=img_count+1

def main(_):
    filename_queue = tf.train.string_input_producer([tfrecord])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/filename': tf.FixedLenFeature([], tf.string),
                                           'image/source_id': tf.FixedLenFeature([], tf.string),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
                                           'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
                                           'image/object/class/text': tf.VarLenFeature(tf.string),
                                           'image/object/class/label': tf.VarLenFeature(tf.int64),
                                       })
    filename = tf.cast(features['image/filename'], tf.string)
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    text = tf.cast(features['image/object/class/text'], tf.string)
    label = tf.cast(features['image/object/class/label'], tf.int64)
    image = tf.image.decode_jpeg(features['image/encoded'])

    labels=[]
    image_ori=[]
    images=None
    boxes_gt_list=[]
    for i in range(batch_size):
        labels.append(label)
        cx = (xmax.values + xmin.values) / 2
        cy = (ymax.values + ymin.values) / 2
        box_w = xmax.values - xmin.values
        box_h = ymax.values - ymin.values
        class_one_hot = tf.one_hot(label.values, class_num)
        box_geo=tf.transpose(tf.stack([cx, cy, box_w, box_h]))
        boxes_info = tf.concat([box_geo, class_one_hot], axis=1)
        boxes_gt_list.append(boxes_info)
        image_ori.append(image)
        image_resized = tf.image.resize_images(image, [image_size, image_size])
        image_resized.set_shape([image_size, image_size, 3])
        image_expanded = tf.expand_dims(image_resized, 0)
        if i==0:
            images=image_expanded
            continue
        images=tf.concat([images,image_expanded],0)

    # with tf.Session() as sess:
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     box_info=sess.run(class_one_hot)
    #     print(box_info)
    #     #print(indices)
    #     # cv2.imshow("Image", dense_mark)
    #     # cv2.waitKey(0)
    #     coord.request_stop()
    #     coord.join(threads)
    global_step = slim.create_global_step()
    with tf.variable_scope('detector'):
        detections, loss = yolo_v3(images, class_num, boxes_gt_list)
        #load_ops = load_weights(tf.global_variables(scope='detector'), weights_file)

    #opt = tf.train.AdamOptimizer(learning_rate=0.1)
    #gradient_all = opt.compute_gradients(loss)
    # var_grad = []
    # for (g, v) in gradient_all:
    #     if 'chamo' in v.name:
    #         var_grad.append((g, v))
    #train_op = opt.apply_gradients(gradient_all)
    # boxes = detections_boxes(detections)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        #sess.run(load_ops)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            #_,loss = sess.run([train_op, loss])
            detected_boxes, loss_np, images1, img_names1= sess.run([detections, loss, image_ori, filename])
            filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=0.5, iou_threshold=0.4)
            classes = load_coco_names(classes_file)
            draw_boxes(filtered_boxes, images1, classes, (image_size, image_size), img_names1)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
