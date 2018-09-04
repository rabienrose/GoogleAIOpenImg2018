import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
img_dir = '/home/chamo/Documents/data/UntitledFolder/test_challenge_2018'
#img_dir='/media/chamo/e9cbf274-e538-4ccc-adbb-16cc0932f014/validation'
out_dir='/home/chamo/Documents/data/UntitledFolder'
class_file='/home/chamo/Documents/work/keras-yolo3/model_data/open_img_code.txt'


def detect_img(yolo):
    categories=[]
    with open(class_file, 'r') as f:
        while True:
            line_str=f.readline()
            if line_str=='':
                break
            categories.append(line_str[0:-1])
    imgs = os.listdir(img_dir)
    with open(out_dir + '/test.csv', 'w') as f:
        re_str = 'ImageId,PredictionString'
        f.write(re_str + '\n')
        count=0
        for img in imgs:
            #print(img)
            count=count+1
            if count%1000==0:
                print(count)
            try:
                image = Image.open(img_dir+'/'+img)
                iw, ih = image.size
            except:
                print('Open Error! Try again!')
                continue
            else:
                re_str=''
                result_dict = yolo.detect_image(image, True)
                re_str = re_str + img.split(".")[0]
                re_str = re_str + ','
                detected_box = result_dict
                for i in range(len(detected_box)):
                    re_str = re_str + categories[result_dict[i]['class']] + ' '
                    re_str = re_str + str(result_dict[i]['conf']) + ' '
                    re_str = re_str + str(detected_box[i][1]/iw) + ' '
                    re_str = re_str + str(detected_box[i][0]/ih) + ' '
                    re_str = re_str + str(detected_box[i][3]/iw) + ' '
                    re_str = re_str + str(detected_box[i][2]/ih) + ' '
                f.write(re_str + '\n')
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
