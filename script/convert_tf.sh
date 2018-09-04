RAW_IMAGES_DIR=/media/chamo/c769deb7-4cd0-4af7-a517-c5601d8313ad/train
OUTPUT_DIR=/media/chamo/c769deb7-4cd0-4af7-a517-c5601d8313ad/train_tfrecord/train
BOUNDING_BOXES=/home/chamo/Documents/data/openImg/train/box
python ../create_oid_tf_record.py \
    --input_box_annotations_csv ${BOUNDING_BOXES}.csv \
    --input_images_directory ${RAW_IMAGES_DIR} \
    --input_label_map ../config/oid_object_detection_challenge_500_label_map.pbtxt \
    --output_tf_record_path_prefix ${OUTPUT_DIR} \
    --num_shards=200
