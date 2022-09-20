from importlib import resources

from sahi.predict import predict,get_prediction,get_sliced_prediction
from sahi.slicing import slice_image

from yolov7.utils.sahiModelExtended import Yolov7DetectionModel

# bgr           :   True if image is read via BGR (defaulted to False as sahi utilise PIL)
# device        :   torch device 
# model_path    :   Path for the model weight
# config_path   :   Path for the model config file
# model_path    :   the type of model being implemented (yolov7/yolov7-e6)
# image_size    :   Inference input size. (defaulted to 640)
# confidence_threshold  :   confidence threshold (defaulted to 0.25)    

weightPath = resources.files('yolov7').joinpath('weights/yolov7-e6_state.pt')
cfgPath = resources.files('yolov7').joinpath('cfg/deploy/yolov7-e6.yaml')

model = Yolov7DetectionModel(
    image_size = 1280,
    model_path = weightPath,
    config_path = cfgPath
)

result = get_sliced_prediction('test.jpg',model)

print("Saving output image into folder")
result.export_visuals(export_dir="sahi_data/")

# sliced_image_result  = slice_image(
#     image = 'test.jpg',
#     # output_file_name = 'output_test_wr_0.2_640',
#     # output_dir = 'sahi_data/',
#     overlap_width_ratio = 0.2,
#     overlap_height_ratio = 0.2,
#     slice_height = 640,
#     slice_width = 640,
#     auto_slice_resolution = False
# )

# for i in sliced_image_result._sliced_image_list:
#     print(i.starting_pixel)
#     print(i.coco_image.height)

print("Completed")