MODEL_ARCH='yolov7' # 'yolov7-tiny', 'yolov7', 'yolov7x', 'yolov7-w6', 'yolov7-e6', 'yolov7-d6', 'yolov7-e6e'
DEPLOY_CFG='/path/to/deploy/cfg/file'
WEIGHT='/path/to/weight'

# Save Paths
REPARAM_WEIGHT='/path/to/save/reparam/weights'
SAVE_PATH='/path/to/final/weights'
NUM_CLASS=1

python reparameterization.py \
  --model_arch $MODEL_ARCH \
  --training_ckpt $WEIGHT \
  --output_ckpt $REPARAM_WEIGHT \
  --deploy_cfg $DEPLOY_CFG \
  --nc $NUM_CLASS

python save_state_dict.py \
  --weights $REPARAM_WEIGHT \
  --save_path $SAVE_PATH \
  --cfg $DEPLOY_CFG