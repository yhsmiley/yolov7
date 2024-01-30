WORKSPACE=/media/data/yolov7
DATA=/media/data/datasets
# DATA2=/media/data/fdet-api

docker run -it --rm \
	--gpus all \
    -w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	-v $DATA:$DATA \
	--shm-size=64g \
	--net host \
	yolov7_main

# -v $DATA2:$DATA2 \
# reid_pipeline_sahi_fr
