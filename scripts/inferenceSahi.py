from sahi.predict import predict,get_prediction
from yolov7.utils.sahiModelExtended import Yolov7DetectionModel
from yolov7.utils.general import increment_path

# model_path    :   the type of model being implemented (yolov7/yolov7-e6)
# image_size    :   the size of the model image (defaulted to 640)
# confidence_threshold  :   confidence threshold (defaulted to 0.25)
model = Yolov7DetectionModel(
    model_path = 'yolov7-e6',
    image_size = 1280
)

print("Getting prediction of the image")
result = get_prediction('test.jpg', model)

print("Saving output image into folder")
result.export_visuals(export_dir="sahi_data/",file_name='outputResult')

# Saving as coco annotations
# prediction_result = result.to_coco_annotations()
# print(prediction_result)

print("Completed")