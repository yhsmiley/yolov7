# converts datasets in COCO format (images in a folder + coco.json) to YOLO format
# python coco_to_yolo.py $IMG_FOLDER $JSON_PATH $OUTPUT_PATH
import click
from turtle import ycor
import cv2
import json
from pathlib import Path
from glob import glob
import shutil

class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:
        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                704
                620
                1401
                1645
            ]
        }
        
    """

    def __init__(self, img_folder, json_path, output_path):
        self.img_folder = img_folder
        self.json_path = json_path
        self.img_output_path = Path(output_path) / "images"
        self.labels_output_path = Path(output_path) / "labels"
        try:
            shutil.rmtree(str(self.labels_output_path))
            shutil.rmtree(str(self.img_output_path))
        except Exception:
            print("Output folders do not exist, creating folders...")
        else:
            print("Deleted existing contents in output folders...")
        self.img_output_path.mkdir(parents=True, exist_ok=True)
        self.labels_output_path.mkdir(parents=True, exist_ok=True)
        

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        try:
            return img.shape
        except AttributeError:
            print('AttributeError for ', img_path)
            return (None, None, None)
    
    def normalise_bbox(self, bbox, img_width, img_height):
        x, y, w, h = bbox
        if x < 0 or x + w > img_width or y < 0 or y + h > img_height:
            return None
        if w > img_width or h > img_height:
            return None
        xc = x + (w / 2.0)
        yc = y + (h / 2.0)
        xc /= img_width
        w /= img_width
        yc /= img_height
        h /= img_height
        return (xc, yc, w, h)

    def convert(self,annotation_key='annotations',imgs_key='images',img_id='image_id',cat_id='category_id',bbox='bbox'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))

        check_set = set()
        img_mappings = {}
        img_dimensions = {}
        for img in data[imgs_key]:
          img_mappings[img["id"]] = img["file_name"]
          img_dimensions[img["id"]] = (img["width"], img["height"])

        # Retrieve data
        for i in range(len(data[annotation_key])):

            # Get required data
            image_id = data[annotation_key][i][img_id]
            image_filename = img_mappings[image_id]
            category_id = f'{data[annotation_key][i][cat_id]}'
            bbox = data[annotation_key][i]['bbox']

            # Convert the data
            yolo_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            img_width, img_height = img_dimensions[image_id]
            normalised_yolo_bbox = self.normalise_bbox(yolo_bbox, float(img_width), float(img_height))

            if not normalised_yolo_bbox:
                continue
            
            # Prepare for export
            filename = f"{Path(image_filename).stem}.txt"
            filepath = Path(self.labels_output_path) / filename
            
            content =f"{category_id} {normalised_yolo_bbox[0]} {normalised_yolo_bbox[1]} {normalised_yolo_bbox[2]} {normalised_yolo_bbox[3]}"

            # Export 
            if image_id in check_set:
                # Append to existing file as there can be more than one label in each image
                file = open(filepath, "a")
                file.write("\n")
                file.write(content)
                file.close()

            elif image_id not in check_set:
                check_set.add(image_id)
                # Write files
                file = open(filepath, "w")
                file.write(content)
                file.close()

    def copy_images(self):
        img_exts = [".jpg", ".jpeg", ".png"]
        for img_label in glob(str(self.labels_output_path) + "/*.txt"):
            img = Path(img_label).stem
            for ext in img_exts:
                img_filepath = Path(self.img_folder) / (img + ext)
                op_filepath = self.img_output_path / Path(img_filepath).name
                try:
                    shutil.copy(img_filepath, str(op_filepath))
                except FileNotFoundError:
                    continue
                else:
                    # Found the file, copied it over
                    break
            else:
                # No img files found
                print(f"WARNING: Unable to find corresponding image for label {img_label}")

    def run(self):
        print("Converting COCO annotations to YOLO's format...")
        self.convert()
        print("Done converting. Copying images...")
        self.copy_images()
        print("Success!")


@click.command()
@click.argument('img_folder')
@click.argument('json_path')
@click.argument('output_path')
def main(img_folder, json_path, output_path):
    ConvertCOCOToYOLO(
      img_folder=img_folder,
      json_path=json_path,
      output_path=output_path
    ).run()
    # ConvertCOCOToYOLO(
    #   img_folder='/home/wenyi/DATA/synthetics/DOTA_GTAV_Experiment/dotav2_3_hbb/',
    #   json_path='/home/wenyi/DATA/synthetics/DOTA_GTAV_Experiment/dotav2_3_hbb/annotations.json',
    #   output_path='/home/wenyi/DATA/synthetics/DOTA_GTAV_Experiment/yolo_dotav2_3_no_clf/'
    # ).run()

if __name__ == "__main__":
    main()
