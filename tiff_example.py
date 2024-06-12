import gdal
import os
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root_dir", default="/app/for_hakaton/Sitronics", required=False)
parser.add_argument("-i", "--img_dir", default="layouts", required=False)
parser.add_argument("-d", "--dest_dir", default="result", required=False) 


import os

def geotiff_to_png(input_path, output_path=None, return_object=False):
    # Open input file
    dataset = gdal.Open(input_path)
    output_types = [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Float32]
    
    # Define output format and options
    options = gdal.TranslateOptions(format='PNG', bandList=[3,2,1], creationOptions=['WORLDFILE=YES'], outputType=output_types[0])
    
    # Translate to PNG
    if output_path is not None:
        gdal.Translate(output_path, dataset, options=options)
        print(f'Successfully saved PNG file to {output_path}')
    
    # Return PNG object
    if return_object:
        mem_driver = gdal.GetDriverByName('MEM')
        mem_dataset = mem_driver.CreateCopy('', dataset, 0)
        png_data = mem_dataset.ReadAsArray()
        return png_data

if __name__ == "__main__":
    args = parser.parse_args()

    root_dir = args.root_dir
    tifs_dir = os.path.join(root_dir, args.img_dir)
    result_dir = os.path.join(root_dir, args.dest_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    tif_names = os.listdir(tifs_dir)
    for name in tqdm(tif_names):
        tif_file = os.path.join(tifs_dir, name)
        dataset = gdal.Open(tif_file)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        cur_shape = array.shape

        geotransform = dataset.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]
        print(f"{name} x_min: {x_min}, y_max: {y_max}, width: {width}, height: {height}, pixel_width: {pixel_width}, pixel_height: {pixel_height}")
