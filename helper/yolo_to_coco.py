# python yolo_to_coco.py --path /home/wenyi/DATA/synthetics/yolo_dota_train_vehs_TILED/ --output test.json --numerical-img-id False
from pathlib import Path

import cv2
import argparse
import json
import numpy as np
import imagesize

#################################################
# Change the classes depend on your own dataset.#
# Don't change the list name 'Classes'          #
#################################################

classes = [
    "vehicle"
]

# Create the annotations of the ECP dataset (Coco format)
coco_format = { "categories": [], "images": [{}],"annotations": [{}]}

def create_image_annotation(file_path: Path, width: int, height: int, image_id: str):
    file_path = file_path.name
    image_annotation = {
        "id": image_id,
        "height": height,
        "width": width,
        "filename": Path(file_path).stem,
    }
    return image_annotation


def create_annotation_from_yolo_format(min_x, min_y, width, height, image_id, category_id, annotation_id):
    bbox = (float(min_x), float(min_y), float(width), float(height))
    area = width * height
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": 0,
        "category_id": category_id
    }

    return annotation

def get_images_info_and_annotations(opt):
    path = Path(opt.path)
    annotations = []
    images_annotations = []
    if path.is_dir():
        image_filepath = path / "images"
        file_paths = sorted(image_filepath.rglob("*.jpg"))
        file_paths += sorted(image_filepath.rglob("*.jpeg"))
        file_paths += sorted(image_filepath.rglob("*.png"))
    else:
        with open(path, "r") as fp:
            read_lines = fp.readlines()
        file_paths = [Path(line.replace("\n", "")) for line in read_lines]

    image_id = 0
    annotation_id = 0

    for file_path in file_paths:
        # Check how many items have progressed
        print("\rProcessing " + str(image_id) + " ...", end='')

        # Build image annotation, known the image's width and height
        coco_image_id = Path(file_path).stem
        if opt.numerical_img_id:
            coco_image_id = str(image_id)

        w, h = imagesize.get(str(file_path))
        image_annotation = create_image_annotation(
            file_path=file_path, width=w, height=h, image_id=coco_image_id
        )
        images_annotations.append(image_annotation)

        label_file_name = f"{file_path.stem}.txt"
        annotations_path = file_path.parents[1] / "labels" / label_file_name

        if not annotations_path.exists():
            continue  # The image may not have any applicable annotation txt file.

        with open(str(annotations_path), "r") as label_file:
            label_read_line = label_file.readlines()

        # yolo format - (class_id, x_center, y_center, width, height)
        # coco format - (annotation_id, x_upper_left, y_upper_left, width, height)
        for line1 in label_read_line:
            label_line = line1
            category_id = (
                int(label_line.split()[0])
            )  # you start with category id with '0'
            x_center = float(label_line.split()[1])
            y_center = float(label_line.split()[2])
            width = float(label_line.split()[3])
            height = float(label_line.split()[4])

            float_x_center = w * x_center
            float_y_center = h * y_center
            float_width = w * width
            float_height = h * height

            min_x = int(float_x_center - float_width / 2)
            min_y = int(float_y_center - float_height / 2)
            width = int(float_width)
            height = int(float_height)


            annotation = create_annotation_from_yolo_format(
                min_x,
                min_y,
                width,
                height,
                coco_image_id,
                category_id,
                annotation_id
            )
            annotations.append(annotation)
            annotation_id += 1

        image_id += 1  # if you finished annotation work, updates the image id.

    return images_annotations, annotations

def get_args():
    parser = argparse.ArgumentParser("YOLO format annotations to COCO dataset format")
    parser.add_argument(
      "-p",
      "--path",
      type=str,
      help="Absolute path for 'train.txt' or 'test.txt', or the root dir for images.",
    )
    parser.add_argument(
      "-o",
      "--output",
      default="./train_coco.json",
      type=str,
      help="Path of the the output json file",
    )
    parser.add_argument(
      "--numerical-img-id",
      default=False,
      type=bool,
      help="Whether the image ID should be numerical. If false, image ID will be the filename",
    )
    args = parser.parse_args()
    return args


def main(opt):
    output_path = opt.output

    print("Start!")

    (
        coco_format["images"],
        coco_format["annotations"],
    ) = get_images_info_and_annotations(opt)

    for index, label in enumerate(classes):
        categories = {
            "supercategory": "NIL",
            "id": index,  # ID starts with '0' .
            "name": label,
        }
        coco_format["categories"].append(categories)

    with open(output_path, "w") as outfile:
        json.dump(coco_format, outfile, indent=4)

    print("\nFinished!")


if __name__ == "__main__":
    options = get_args()
    main(options)
