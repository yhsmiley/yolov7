# python tiling.py /path/to/yolo-format/folder /path/to/output/folder
# eg. python tiling.py /home/wenyi/DATA/synthetics/DOTA_yolo_small_vehs_only/ /home/wenyi/DATA/synthetics/DOTA_yolo_small_vehs_only_TILED/

from pathlib import Path
import click
import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
import os

def tile_images(dir, op_dir, slice_size, falsepath):
    img_path = str(dir / "images")
    labels_path = str(dir / "labels")
    new_img_path = str(op_dir / "images")
    new_labels_path = str(op_dir / "labels")
    Path(new_img_path).mkdir(parents=True, exist_ok=True)
    Path(new_labels_path).mkdir(parents=True, exist_ok=True)

    # tile all images in a loop
    for t, img_filename in enumerate(os.listdir(img_path)):
        if t % 100 == 0:
            print(f"=== {t} out of {len(os.listdir(img_path))}")

        imname, imext = os.path.splitext(img_filename)
        try:
            im = Image.open(os.path.join(img_path, img_filename))
        except Image.DecompressionBombError:
            print(f"DecompressionBombError. Skipping image...")
            continue
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = os.path.join(labels_path, f'{imname}.txt')
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
        
        # we need to rescale coordinates from 0-1 to real image height and width
        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height
        
        boxes = []
        
        # convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w']/2
            y1 = (height - row[1]['y1']) - row[1]['h']/2
            x2 = row[1]['x1'] + row[1]['w']/2
            y2 = (height - row[1]['y1']) + row[1]['h']/2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
        

        # create tiles and find intersection with bounding boxes for each tile
        for i in range((height // slice_size)):
            for j in range((width // slice_size)):
                x1 = j*slice_size
                y1 = height - (i*slice_size)
                x2 = ((j+1)*slice_size) - 1
                y2 = (height - (i+1)*slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])        
                        
                        if not imsaved:
                            sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                            sliced_im = Image.fromarray(sliced)
                            rgb_im = sliced_im.convert('RGB')
                            slice_path = os.path.join(new_img_path, f'{imname}_{i}_{j}{imext}')
                            
                            slice_labels_path = os.path.join(new_labels_path, f'{imname}_{i}_{j}.txt')
                            
                            rgb_im.save(slice_path)
                            imsaved = True                    
                        
                        # get the smallest polygon (with sides parallel to the coordinate axes) that contains the intersection
                        new_box = inter.envelope 
                        
                        # get central point for the new bounding box 
                        centre = new_box.centroid
                        
                        # get coordinates of polygon vertices
                        try:
                            x, y = new_box.exterior.coords.xy
                        except AttributeError:
                            print(f"AttributeError in: {img_filename}")
                            continue
                        
                        # get bounding box width and height normalized to slice size
                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size
                        
                        # we have to normalize central x and invert y for yolo format
                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])
                
                # save txt with labels for the current tile
                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
                
                # if falsepath is indicated & there are no bounding boxes intersect current tile, save this tile to a separate folder
                if falsepath and not imsaved:
                    sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
                    sliced_im = Image.fromarray(sliced)
                    rgb_im = sliced_im.convert('RGB')
                    slice_path = os.path.join(falsepath, f'{imname}_{i}_{j}{imext}')                

                    rgb_im.save(slice_path)
                    imsaved = True


@click.command()
@click.argument('dataset_directory')
@click.argument('output_folder')
@click.option('-f', '--falsepath')
@click.option('--size', default=640)
def main(dataset_directory, output_folder, falsepath, size):
    # printing config
    print(f"slice size: {size}")
    if not falsepath:
        print("no falsepath indicated. empty tiles will be thrown away")

    if falsepath is not None and not os.path.isdir(falsepath):
        Path(falsepath).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_folder)
    if (Path(dataset_directory) / "images").is_dir() and (Path(dataset_directory) / "labels").is_dir():
        if not output_path.is_dir():
            print(f"output folder {str(output_path)} does not exist. creating...")
            output_path.mkdir(parents=True, exist_ok=True)
        print(f"tiling '{dataset_directory}' into '{output_path}'")
        tile_images(Path(dataset_directory), output_path, size, falsepath)
    else:
        sub_folders = [x for x in Path(dataset_directory).iterdir() if x.is_dir()]
        for i, sub_folder in enumerate(sub_folders):
            if (sub_folder / "images").is_dir() and (sub_folder / "labels").is_dir():
                op_folder = output_path / sub_folder.name
                if not op_folder.is_dir():
                    print(f"output folder {str(op_folder)} does not exist. creating...")
                    op_folder.mkdir(parents=True, exist_ok=True)
                print(f"{i + 1} of {len(sub_folders)}: tiling '{sub_folder}' into '{op_folder}'")
                tile_images(sub_folder, op_folder, size, falsepath)
            else:
                print(f"cannot find 'images' or 'labels' folder in {sub_folder}. skipping tiling this folder...")
    
    print("completed")

if __name__ == "__main__":
    main()
