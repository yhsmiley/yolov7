# python combine_yolo_datasets.py -folders /path/here /path/here -output-folder /path/here --img-type .jpg
from glob import glob
from pathlib import Path
import shutil
import argparse

def main(folders, output_folder, img_type):
  if folders is None or output_folder is None:
    print("Input & output folders cannot be None")
    return

  output_images_folder = output_folder / "images"
  output_labels_folder = output_folder / "labels"
  output_images_folder.mkdir(parents=True, exist_ok=True)
  output_labels_folder.mkdir(parents=True, exist_ok=True)

  for i, folder in enumerate(folders):
    images_folder = folder / "images"
    labels_folder = folder / "labels"
    
    for image in glob(str(images_folder) + "/*" + img_type):
      dest = str(i) + "_" + Path(image).name
      shutil.copy(image, str(output_images_folder / dest))
    
    for label in glob(str(labels_folder) + "/*.txt"):
      dest = str(i) + "_" + Path(label).name
      shutil.copy(label, str(output_labels_folder / dest))
    print("\r " + str(i + 1) + " out of " + str(len(folders)) + " completed", end="")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-folders', nargs='+', type=Path, default=None, help='yolo-formatted dataset folders to merge')
  parser.add_argument('-output-folder', type=Path, default=None, help='output folder for the merged dataset')
  parser.add_argument('--img-type', type=str, default='.jpg', help='image type (eg. png, jpg, jpeg)')
  opt = parser.parse_args()
  main(opt.folders, opt.output_folder, opt.img_type)
