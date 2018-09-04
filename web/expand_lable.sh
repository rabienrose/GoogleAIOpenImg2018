HIERARCHY_FILE=/home/chamo/Documents/data/openImg/bbox_labels_600_hierarchy.json
BOUNDING_BOXES=/home/chamo/Documents/data/openImg/val/box
IMAGE_LABELS=/path/to/challenge-2018-train-annotations-human-imagelabels
export PYTHONPATH=$PYTHONPATH:/output/models/research:/output/models/research/slim
python ./oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1
