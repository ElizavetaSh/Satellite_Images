import gdal
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root_dir", default="databases/vadim_6_komi_gt_pred", required=False)
parser.add_argument("-m", "--masks_dir", default="new_masks", required=False)
parser.add_argument("-i", "--img_dir", default="images", required=False)
parser.add_argument("-d", "--dest_dir", default="geo_komi_polygons_without_points", required=False)

if __name__ == "__main__":
    args = parser.parse_args()

    root_dir = args.root_dir
    masks_dir = os.path.join(root_dir, args.masks_dir)
    tifs_dir = os.path.join(root_dir, args.img_dir)
    result_dir = os.path.join(root_dir, args.dest_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    tif_names = os.listdir(tifs_dir)

    for name in tqdm(tif_names):
        tif_file = os.path.join(tifs_dir, name)
        dataset = gdal.Open(tif_file)
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()

        geotransform = dataset.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        png_file = os.path.join(masks_dir, name[:-4]+".png")
        png_dataset = gdal.Open(png_file)
        driver = gdal.GetDriverByName('GTiff')
        output_file = os.path.join(result_dir, name)
        output_dataset = driver.CreateCopy(output_file, png_dataset)
        output_dataset.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, pixel_height))

        output_dataset = None
        dataset = None
        png_dataset = None
