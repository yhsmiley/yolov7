# Helper Functions

## Dataset Functions
### Combining Datasets
Combines datasets of the same annotation format together

Relevant scripts:
- `combine_yolo_datasets.py`
- For combining COCO datasets, you may use [pyodi](https://gradiant.github.io/pyodi/reference/apps/coco-merge/). For example:
```bash
pyodi coco merge coco_1.json coco_2.json output.json
```

### Converting Datasets with Different Annotation Formats
Converts datasets between COCO and YOLO format

*Note*: YOLO annotation format consists of a `.txt` file for each image in the same directory. Each `.txt` file contains the annotations for the corresponding image file, with each line indicating 1 bounding box:
```<object-class> <xc> <yc> <width> <height>```
* `xc` and `yc`: The X and Y coordinates of the object's center point within the image, normalized to be between 0 and 1.

Each image must be of the same size, `640x640` or `1280x1280` are preferred, although rectangle images can be used too. In the case where:
- images are bigger than the specified size, `tiling.py` can be used.
- images are smaller than the specified size, `padding.py` can be used.


Relevant scripts (COCO to YOLO):
- `coco_to_yolo.py` / `run_coco_to_yolo.sh`
- `tiling.py`
- `split_train_val_test.py`

Relevant scripts(YOLO to COCO)
- `yolo_to_coco.py`

### Visualising YOLO Datasets
Visualise a YOLO dataset image and its annotations

Relevant script:
- `yolo_annotator.py`

## Evaluation Functions
### Evaluating F-Beta scores
To evaluate F-Beta scores (F1-Score & F2-Score), you can do it individually (`evaluate_fbeta.py`) or in a batch (`batch_evaluate_fbeta.py`).
Currently, the batch function evaluates slices of 50 to 1000 with a step size of 50

This function requires the generated `prediction.json` and the ground-truth json, both in COCO annotation format.

To generate the `prediction.json` in yolov7, simply add `--save-json` as a parameter when running `test.py`, or use `run_batch_test.sh`/`run_test.sh` in this repo

Relevant scripts:
- `evaluate_fbeta.py` / `run_evaluate_fbeta.sh`
- `batch_evaluate_fbeta.py`