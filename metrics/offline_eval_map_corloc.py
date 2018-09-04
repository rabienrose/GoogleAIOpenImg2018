import csv
import os
import tensorflow as tf
import utils.object_detection_evaluation
from utils import label_map_util
import numpy as np
from core import standard_fields

eval_dir = '/home/chamo/Documents/data/openImg/val'
detected_re_path = '/home/chamo/Documents/data/UntitledFolder/try.csv'
gt_path = '/home/chamo/Documents/data/openImg/val/box.csv'
label_map_path='/home/chamo/Documents/work/OpenImgChamo/config/oid_object_detection_challenge_500_label_map.pbtxt'

def read_data_and_evaluate():
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)
    object_detection_evaluator = utils.object_detection_evaluation.OpenImagesDetectionChallengeEvaluator(categories)
    with open(detected_re_path, "r") as detected_input:
        line = f_result.readline()
        while True:
            line = f_result.readline()
            if line == '':
                break
            line_count = line_count + 1
            splited = line.split(",")
            image_name = splited[0]
            if image_name in result_dict.keys():
                print('rep img')
            if len(splited) < 2:
                print('error')
            if splited[1] != '\n':
                cache_re = splited[1]
                splited = splited[1].split(' ')
                box_count = 0
                new_str = ''
                while True:
                    new_str = new_str + splited[box_count * 6 + 0] + ' '
                    new_str = new_str + splited[box_count * 6 + 1] + ' '
                    new_str = new_str + splited[box_count * 6 + 3] + ' '
                    new_str = new_str + splited[box_count * 6 + 2] + ' '
                    new_str = new_str + splited[box_count * 6 + 5] + ' '
                    new_str = new_str + splited[box_count * 6 + 4] + ' '
                    box_count = box_count + 1
                    if len(splited) - 1 <= box_count * 6:
                        break
                result_dict[image_name] = new_str + '\n'
                img_count = img_count + 1


    with open(gt_path, "r") as gt_input:
        line = gt_input.readline()
        last_img_name = ''
        while True:
            count = count + 1
            line = gt_input.readline()
            if line == '':
                break
            splited_str = line.split(",")
            xmin=float(splited_str[4])
            xmax = float(splited_str[5])
            ymin = float(splited_str[6])
            ymax = float(splited_str[7])
            class_text=str(splited_str[2])
            if last_img_name == splited_str[0]:
                boxes=np.concatenate(boxes,[xmin, xmax, ymin, ymax], axis=0)
                classes.append(class_text)
            else:
                gt_dict={}
                gt_dict[standard_fields.InputDataFields.groundtruth_boxes]=np.transpose(boxes)
                gt_dict[standard_fields.InputDataFields.groundtruth_classes]=classes
                object_detection_evaluator.add_single_ground_truth_image_info(last_img_name, gt_dict)
                last_img_name = splited_str[0]
                boxes = [[xmin, xmax, ymin, ymax]]
                classes = class_text

    return object_detection_evaluator.evaluate()

def write_metrics(metrics, output_dir):
    tf.logging.info('Writing metrics.')

    with open(os.path.join(output_dir, 'metrics.csv'), 'w') as csvfile:
        metrics_writer = csv.writer(csvfile, delimiter=',')
        for metric_name, metric_value in metrics.items():
            metrics_writer.writerow([metric_name, str(metric_value)])


def main(argv):
    metrics = read_data_and_evaluate()
    write_metrics(metrics, eval_dir)


if __name__ == '__main__':
    tf.app.run(main)
