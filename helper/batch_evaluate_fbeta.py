# Using Yinghui's version https://github.com/yhsmiley/fdet-api/. Please follow the set up guide.
# python evaluate_fbeta.py your_prefix /path/to/pred/folder /path/to/groundtruth/json /path/to/output/csv
# eg. python evaluate_fbeta.py dota_veh_finetune /home/DATA/yolov7/runs/test /home/DATA/groundtruth.json /home/DATA/output/dota_veh_finetune_results.csv

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from glob import glob
import click
import csv

def evaluate_fbeta(true_path, pred_path):
  cocoGt = COCO(true_path)  # initialize COCO ground truth api
  cocoDt = cocoGt.loadRes(pred_path)  # initialize COCO prediction api
  cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # initialize COCO evaluation api
  cocoEval.evaluate()
  cocoEval.accumulateFBeta()
  f1_score, _, precision, recall = cocoEval.getBestFBeta(beta=1, iouThr=.65, average='macro')
  f2_score, _, precision, recall = cocoEval.getBestFBeta(beta=2, iouThr=.65, average='macro')
  return f1_score, f2_score, precision, recall

# pycocotools require each image to have at least 1 bbox for evaluation
# this function fills in images with no annotations with a dummy annotation (edits the prediction.json file directly)
def check_and_fill_in_missing_annotations(pred_json, gt_file):
  with open(pred_json, 'r') as pred_f, open(gt_file, 'r') as gt_f:
    data_true = json.load(gt_f)
    data_pred = json.load(pred_f)

  pred_imgs_set = set()
  for pred in data_pred:
    pred_imgs_set.add(pred['image_id'])
  
  gt_imgs_set = set()
  for img in data_true['images']:
    gt_imgs_set.add(img['id'])
  
  missing_images = gt_imgs_set - pred_imgs_set
  for missing_image in missing_images:
    # pad images with no annotations with a bounding box of [0, 0, 0, 0]
    data_pred.append({
      "image_id": missing_image,
      "category_id": 0,
      "bbox": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "score": 0.0
    })

  # replace prediction json file
  if len(missing_images) > 0:
    with open(pred_json, 'w') as pred_f:
      json.dump(data_pred, pred_f)

# runs fbeta evaluation for a batch of 20 runs (50 to 1000 with step size 50)
# saves results (f1, f2, precision, recall) of the batch into .csv file
def run_fbeta_evaluation(pred_folder, prefix, gt_file, output_csv):
  print("running fbeta eval")
  results = []
  error_folders = []
  for i in range(50, 1001, 50): # edit here if batch is different
    folder_name = prefix + "_" + str(i)
    pred_path = Path(pred_folder) / folder_name
    pred_json = glob(str(pred_path) + "/*.json")
    if len(pred_json) != 1:
      error_folders.append(str(pred_path))
      continue
    pred_json = pred_json[0]
    check_and_fill_in_missing_annotations(pred_json, gt_file)
    f1_score, f2_score, precision, recall = evaluate_fbeta(gt_file, pred_json)
    results.append([i, f1_score, f2_score, precision, recall])

  with open(output_csv, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    header = ['images', 'f1', 'f2', 'precision', 'recall']
    writer.writerow(header)
    for result in results:
      writer.writerow(result)

  if len(error_folders) > 0:
    print("\nERROR: Unable to evaluate f-beta for the following folders because there are 0 or too many prediction.json files.\nAffected folders: ", error_folders)
  
  print("completed")

@click.command()
@click.argument('pred_folder')
@click.argument('prefix')
@click.argument('gt_file')
@click.argument('output_csv')
def main(pred_folder, prefix, gt_file, output_csv):
  run_fbeta_evaluation(pred_folder, prefix, gt_file, output_csv)


if __name__ == '__main__':
  main()
