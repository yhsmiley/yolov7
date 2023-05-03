# Using Yinghui's version https://github.com/yhsmiley/fdet-api/. Please follow the set up guide.
# results are printed onto console
# eg. python evaluate_fbeta.py /home/wenyi/DATA/yolov7/runs/test/vedai_proxy_veh/yolov7_synthgtav_proxy_vehs_640_bs32_e200_predictions.json /home/wenyi/DATA/VEDAI/vedai-veh.json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from glob import glob
import click

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
  print(pred_json, gt_file)
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

# prints f1 and f2 scores, together with precision and recall
def run_fbeta_evaluation(pred_json, gt_file):
  print("running fbeta eval")
  pred_json = pred_json
  check_and_fill_in_missing_annotations(pred_json, gt_file)
  f1_score, f2_score, precision, recall = evaluate_fbeta(gt_file, pred_json)
  print("f1_score, f2_score, precision, recall")
  print(f1_score, f2_score, precision, recall)

  print("completed")

@click.command()
@click.argument('pred_json')
@click.argument('gt_file')
def main(pred_json, gt_file):
  run_fbeta_evaluation(pred_json, gt_file)

if __name__ == '__main__':
  main()
