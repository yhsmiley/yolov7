# docker build -t yolov7_main .

FROM nvcr.io/nvidia/pytorch:23.12-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/home/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

# RUN apt-get -y update && \
#     apt-get -y upgrade && \
#     apt -y update && \
#     apt-get install --no-install-recommends -y \
#         software-properties-common \
#         build-essential \
#         gpg-agent \
#         pkg-config \
#         git

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

### APT END ###

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install cython && \
    pip3 install 'git+https://github.com/yhsmiley/fdet-api.git#subdirectory=PythonAPI'

# python3 train.py --workers 8 --device 0 --batch-size 8 --data data/coco128.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'weights/orig/yolov7_training.pt' --name yolov7-py --hyp data/hyp.scratch.p5.yaml --exist-ok --epochs 10 --freeze 50
