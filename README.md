## Google AI Open Images - Object Detection Track
https://www.kaggle.com/c/google-ai-open-images-object-detection-track

### File description
  * Training model （Faster and SSD）: train_frcnn.py
  * Training model （Yolo）: train_yolo.py
  * Generate the submission csv: eval.py
  * Evaluate with yolo: yolo_eval_csv.py
  * Convert raw img to tfrecord: create_oid_tf_record.py
  * Visual check the submission: vis_infer_result.py
  * Visual check tfrecord file: check_tfrecord.py

### Working folder description
* script: the bash files to run all the app
* config: config files needed for app
* All other main files should placed in the root folder
* web: a simple framework to run a server app that can recogization simgle image.

### Data file description
* tfrecord files: 
* box.csv: ground truth of box and classes info for each image
* oid_object_detection_challenge_500_label_map.pbtxt: mapping between machine code of class and human readable name
* faster_rcnn.config: config files for tfrecord, checkpoint

### Download dataset
  * https://github.com/cvdfoundation/open-images-dataset
  * https://docs.aws.amazon.com/cli/latest/userguide/installing.html
  * https://storage.googleapis.com/openimages/web/download.html
    * Only download the csv from this site
    * The csv files include the box, image class, image URL information

### Install Code
  * sudo /home/chamo/.pyenv/versions/anaconda3-5.1.0/bin/protoc ./object_detection/protos/*.proto --python_out=.

### Requirement
  * tensorflow 1.8
  * Other python package which can be eaily installed by pip

### Members:
* https://www.kaggle.com/rabienrose
* https://www.kaggle.com/bryanbocao
* https://github.com/rabienrose
* https://github.com/BryanBo-Cao/x-lab


### Links
* http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
* https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab
* https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/
* https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

