HIERARCHY_FILE=/home/chamo/Documents/data/openImg/bbox_labels_600_hierarchy.json
BOUNDING_BOXES=/home/chamo/Documents/data/openImg/val/box

python ../oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1


    #--json_hierarchy_file=/home/yiming/Documents/data/val/bbox_labels_500_hierarchy.json --input_annotations=/home/yiming/Documents/data/val/box.csv --output_annotations=/home/yiming/Documents/data/val/box_expanded.csv --annotation_type=1
