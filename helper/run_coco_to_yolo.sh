IMG_FOLDER='/path/to/img/folder'
JSON_PATH='/path/to/annotations/json'
OUTPUT_PATH='path/to/output/non-tiled/folder'
OUTPUT_TILED_PATH='/path/to/output/tiled/folder'
FALSEPATH='/path/to/store/empty/tiled/images'

python coco_to_yolo.py \
  $IMG_FOLDER \
  $JSON_PATH \
  $OUTPUT_PATH

# Optional: Further split into train, val & test folders
# Remember to delete "images" and "labels" from $OUTPUT_PATH
python split_train_val_test.py \
  $OUTPUT_PATH \
  $OUTPUT_PATH \
  --test True

# Optional: Tiling
python tiling.py \
  $OUTPUT_PATH \
  $OUTPUT_TILED_PATH \
  --falsepath $FALSEPATH \
  --size 640