# randomly splits folder into train and val
from glob import glob
from random import shuffle
from pathlib import Path
import shutil
import click

class SplitTrainVal:

  def __init__(self, input_folder, output_folder, train=0.8, test=0.1):
    self.labels_folder = Path(input_folder) / "labels"
    self.images_folder = Path(input_folder) / "images"
    self.output_folder = Path(output_folder)
    self.train = train
    self.test = test

  def split_into_folders(self, labels, folder_name):
    # Create relevant folders
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

    # Copying images into their new folders
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


  def run(self, split_test=False):
    all_labels = glob(str(self.labels_folder) + "/*.txt")
    shuffle(all_labels)

    num_of_images = len(all_labels)
    num_of_train_images = int(self.train * num_of_images)

    if split_test:
      val_ratio = round(1.0 - self.train - self.test, 4) # floating point precision issues
      val_percentage_of_remaining = val_ratio / (val_ratio + self.test)
      # num_of_val_images = int(val_ratio * num_of_images)
      num_of_val_images = int((num_of_images - num_of_train_images) * val_percentage_of_remaining)
    else:
      num_of_val_images = num_of_images - num_of_train_images

    train_labels = all_labels[:num_of_train_images]
    val_labels = all_labels[num_of_train_images:num_of_train_images+num_of_val_images]

    # Split into folders
    self.split_into_folders(train_labels, "train")
    print(f"Done with train folder: {num_of_train_images} images")
    self.split_into_folders(val_labels, "val")
    print(f"Done with val folder: {num_of_val_images} images")

    if split_test:
      test_labels = all_labels[num_of_train_images+num_of_val_images:]
      num_of_test_images = num_of_images - num_of_train_images - num_of_val_images
      self.split_into_folders(test_labels, "test")
      print(f"Done with test folder: {num_of_test_images} images")
      print(f"Final split of {num_of_images} images: {{ Train: {self.train:.2f} ({num_of_train_images}), Val: {val_ratio:.2f} ({num_of_val_images}), Test: {self.test:.2f} ({num_of_test_images}) }}")
    else:
      print(f"Final split of {num_of_images} images: {{ Train: {self.train:.2f} ({num_of_train_images}), Val: {(1.0 - self.train):.2f} ({num_of_val_images}) }}")

@click.command()
@click.argument('input_folder')
@click.argument('output_folder')
@click.option('-t', '--test') # split into test folder as well
def main(input_folder, output_folder, test):
  if test:
    print(f"Splitting into train, val and test folders")
    SplitTrainVal(
      input_folder=input_folder,
      output_folder=output_folder
    ).run(
      split_test=True
      )
  else:
    print(f"Splitting into train and val folders")
    SplitTrainVal(
      input_folder=input_folder,
      output_folder=output_folder
    ).run()
  
  # Clean up: Deleting original `images` and `labels` folders if input_folder == output_folder
  if input_folder == output_folder:
    shutil.rmtree(str((Path(input_folder) / 'images')))
    shutil.rmtree(str((Path(input_folder) / 'labels')))
    print(f"Detected that input and output folders are the same")
    print(f"Deleted existing `images` and `labels` folders in the folder")
  print("Success!")

if __name__ == "__main__":
  main()
