import functools
import os
import tensorflow as tf
import numpy as np
from builders import dataset_builder
from builders import model_builder
from utils import config_util
from utils import dataset_util
from utils import label_map_util
from core import prefetcher
from core import standard_fields as fields
from PIL import Image

tf.logging.set_verbosity(0)

flags = tf.app.flags

flags.DEFINE_string('checkpoint_dir', '/home/chamo/Documents/work/keras-yolo3/logs/000/trained_weights_final.h5',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('imgs_dir', '/media/chamo/e9cbf274-e538-4ccc-adbb-16cc0932f014/validation','')
flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
FLAGS = flags.FLAGS


def main(unused_argv):
    assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
    assert FLAGS.eval_dir, '`eval_dir` is missing.'
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    configs = config_util.get_configs_from_pipeline_file(
        FLAGS.pipeline_config_path)
    tf.gfile.Copy(FLAGS.pipeline_config_path, os.path.join(FLAGS.eval_dir, 'pipeline.config'), overwrite=True)
    model_config = configs['model']
    input_config = configs['eval_input_config']
    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)


    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes, False)

    model = model_fn()
    image_raw_data = tf.placeholder(tf.string, None)
    img_data_jpg = tf.image.decode_jpeg(image_raw_data)
    input_img = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
    input_img.set_shape([None, None, 3])
    original_image = tf.expand_dims(input_img, 0)
    preprocessed_image, true_image_shapes = model.preprocess(tf.to_float(original_image))
    #preprocessed_image = tf.reshape(preprocessed_image, [-1, h, w, 3])
    prediction_dict = model.predict(preprocessed_image, true_image_shapes)
    detections = model.postprocess(prediction_dict, true_image_shapes)
    label_id_offset = 1
    output_dict = {}
    detection_fields = fields.DetectionResultFields
    detection_boxes = detections[detection_fields.detection_boxes][0]
    detection_scores = detections[detection_fields.detection_scores][0]
    detection_classes = (tf.to_int64(detections[detection_fields.detection_classes][0])+label_id_offset)
    num_detections = tf.to_int32(detections[detection_fields.num_detections][0])
    detection_boxes = tf.slice(detection_boxes, begin=[0, 0], size=[num_detections, -1])
    detection_classes = tf.slice(detection_classes, begin=[0], size=[num_detections])
    detection_scores = tf.slice(detection_scores, begin=[0], size=[num_detections])
    output_dict[detection_fields.detection_boxes] = detection_boxes
    output_dict[detection_fields.detection_classes] = detection_classes
    output_dict[detection_fields.detection_scores] = detection_scores
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session('', graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    #latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    latest_checkpoint=FLAGS.checkpoint_dir+'/model.ckpt'
    print("latest_checkpoint:" + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    with open(FLAGS.eval_dir+'/test.csv', 'w') as f:
        re_str='ImageId,PredictionString\n'
        imgs = os.listdir(FLAGS.imgs_dir)
        img_count=0
        for img in imgs:
            img_count=img_count+1
            print(img_count)
            image_raw_data_jpg = tf.gfile.FastGFile(FLAGS.imgs_dir+'/'+img, 'rb').read()
            result_dict = sess.run(output_dict, feed_dict={image_raw_data: image_raw_data_jpg})
            re_str=re_str+img.split(".")[0]
            re_str = re_str + ','
            detected_box=result_dict[detection_fields.detection_boxes]
            for i in range(len(detected_box)):
                #print(result_dict[detection_fields.detection_classes][i]-1)
                #print(categories[0])
                re_str=re_str+categories[result_dict[detection_fields.detection_classes][i]-1]['name']+' '
                re_str = re_str + str(result_dict[detection_fields.detection_scores][i]) + ' '
                re_str = re_str + str(detected_box[i][1]) + ' '
                re_str = re_str + str(detected_box[i][0]) + ' '
                re_str = re_str + str(detected_box[i][3]) + ' '
                re_str = re_str + str(detected_box[i][2]) + ' '
            f.write(re_str+'\n')
            re_str=''
    sess.close()


if __name__ == '__main__':
    tf.app.run()
