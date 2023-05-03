# randomly splits folder into train and val
from glob import glob
from random import shuffle
from pathlib import Path
import shutil
import click

class SplitTrainVal:

  def __init__(self, input_folder, output_folder, train=0.8):
    self.labels_folder = Path(input_folder) / "labels"
    self.images_folder = Path(input_folder) / "images"
    self.output_folder = Path(output_folder)
    self.train = train

  def split_into_folders(self, labels, folder_name):
    op_img_parent_folder = self.output_folder / folder_name / "images"
    op_label_parent_folder = self.output_folder / folder_name / "labels"
    print(f"Output will be stored in:\nImages: {op_img_parent_folder}\nLabels:{op_label_parent_folder}")

    try:
        shutil.rmtree(str(op_img_parent_folder))
        shutil.rmtree(str(op_label_parent_folder))
    except Exception:
        print("Output folders do not exist, creating folders...")
    else:
        print("Deleted existing contents in output folders...")
    op_img_parent_folder.mkdir(parents=True, exist_ok=True)
    op_label_parent_folder.mkdir(parents=True, exist_ok=True)

    img_exts = [".jpg", ".jpeg", ".png"]
    for label in labels:
      label_path = Path(label)
      for ext in img_exts:
        image_filename = self.images_folder / (label_path.stem + ext)
        if Path.exists(image_filename):
          break
      else:
        print(f"WARNING: Unable to find corresponding image for label {label}")
        continue
      image_dst = op_img_parent_folder / image_filename.name
      label_dst = op_label_parent_folder / label_path.name
      shutil.copy(str(image_filename), str(image_dst))
      shutil.copy(str(label), str(label_dst))


  def run(self):
    all_labels = glob(str(self.labels_folder) + "/*.txt")
    num_of_images = len(all_labels)
    num_of_train_images = int(self.train * num_of_images)
    shuffle(all_labels)
    train_labels = all_labels[:num_of_train_images]
    val_labels = all_labels[num_of_train_images+1:]

    # Split into folders
    self.split_into_folders(train_labels, "train")
    print(f"Done with train folder: {num_of_train_images} images")
    self.split_into_folders(val_labels, "val")
    print(f"Done with val folder: {num_of_images - num_of_train_images} images")

@click.command()
@click.argument('input_folder')
@click.argument('output_folder')
def main(input_folder, output_folder):
  SplitTrainVal(
    input_folder=input_folder,
    output_folder=output_folder
  ).run()
  print("Success!")

if __name__ == "__main__":
  main()
