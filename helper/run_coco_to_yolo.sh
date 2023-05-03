IMG_FOLDER='/path/to/img/folder'
JSON_PATH='/path/to/annotations/json'
OUTPUT_PATH='path/to/output/non-tiled/folder'
OUTPUT_TILED_PATH='/path/to/output/tiled/folder'

python coco_to_yolo.py $IMG_FOLDER $JSON_PATH $OUTPUT_PATH
# Optional: Further split into train & val folders
python split_train_val.py $OUTPUT_PATH $OUTPUT_PATH # Remember to delete "images" and "labels" from $OUTPUT_PATH
# Optional: Tiling
# python tiling.py $OUTPUT_PATH $OUTPUT_TILED_PATH --size 640