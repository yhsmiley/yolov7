## Training with COCO128 dataset

1. Build the docker image: `docker build -t yolov7_main .` OR load with `docker load -i yolov7_main.tar.gz`
1. Go into the docker container: `bash run_docker.sh`
1. Download pretrained weights from [here](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#transfer-learning) and put into `weights` folder
1. Run the training script: `python3 train.py --workers 8 --device 0 --batch-size 8 --data data/coco128.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'weights/yolov7_training.pt' --name <name> --hyp data/hyp.scratch.p5.yaml --exist-ok --epochs 100`
1. Results will be stored in `runs/train/<name>` folder

## Testing with COCO128 dataset

1. Build the docker image: `docker build -t yolov7_main .` OR load with `docker load -i yolov7_main.tar.gz`
1. Go into the docker container: `bash run_docker.sh`
1. Run the testing script: `python test_coco.py --weights weights/yolov7_state.pt --cfg cfg/deploy/yolov7.yaml --data data/coco128.yaml --batch-size 32 --img-size 1280 --conf-thres 0.001 --iou-thres 0.5 --task test --device 0 --save-json --evaluate-fbeta`
1. Results will be stored in `runs/test/exp<num>` folder
```
Sample Results:
       best f1-score        best f2-score                  map                map50
  0.7176115159054559   0.6951397249857424   0.3747956051755414    0.704324253037085
```

### Things to change

1. `run_docker.sh`: change `WORKSPACE` to the yolov7 repo location, and `DATA` to the dataset location
1. `data/coco128.yaml`: change `train` `val` `test` to the correct location

### Performance Metrics

1. F1 score (`best f1-score` in results)
1. Mean Average Precision (`map50` in results)
