RAW_IMAGES_DIR=/media/chamo/chamo/googleImg/validation
OUTPUT_DIR=/media/chamo/chamo/googleImg/val_tfrecord
BOUNDING_BOXES=/home/chamo/Documents/data/openImg/val/box
export PYTHONPATH=$PYTHONPATH:/output/models/research:/output/models/research/slim
python ./create_oid_tf_record.py \
    --input_box_annotations_csv ${BOUNDING_BOXES}_expanded.csv \
    --input_images_directory ${RAW_IMAGES_DIR} \
    --input_label_map /output/models/research/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
    --output_tf_record_path_prefix ${OUTPUT_DIR} \
    --num_shards=100
