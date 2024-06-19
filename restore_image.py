import argparse
import os

import pandas as pd
import numpy as np

import gdal



def restore_by_mean(channel, i, j):
    height, width = channel.shape
    neighbors = [
        channel[i-1, j] if i > 0 else 0,
        channel[i+1, j] if i < height-1 else 0,
        channel[i, j-1] if j > 0 else 0,
        channel[i, j+1] if j < width-1 else 0
    ]
    avg = np.mean(neighbors[neighbors != 0])
    return avg



def restore_pixels_4channel(img, restore_method):
    """
    Восстанавливает "битые" пиксели на 4-канальном изображении (R, G, B, NIR).

    Args:
        img: 4-канальное изображение в формате (R, G, B, NIR).

    Returns:
        Восстановленное изображение.
    """

    # Функция для поиска и восстановления битых пикселей в канале
    def restore_channel(channel, num_ch):
        x = np.sort(channel.reshape(-1))
        q75, q25 = np.percentile(x, [75 ,25])
        iqr = q75 - q25
        max_v = q75 + 1.5 * iqr
        min_v = q25 - 1.5 * iqr
        bad_pixels = []
        height, width = channel.shape
        for i in range(height):
            for j in range(width):
                if channel[i, j] == 0 or channel[i, j] > 2.5 * max_v or channel[i, j] < 0.5 * min_v:
                    bad_v = channel[i, j]
                    restored_pixel = restore_method(channel, i, j)
                    channel[i, j] = restored_pixel
                    bad_pixels.append([i, j, num_ch, bad_v, restored_pixel])
        return channel, bad_pixels

    restored_img = []
    bad_pixels = []
    for i, channel in enumerate(img):
        restored_channel, bad_pixels = restore_channel(channel, i)
        restored_img.append(restored_channel)
        bad_pixels.extend(bad_pixels)


    restored_img = np.stack(restored_img, 0)

    return restored_img, bad_pixels


def save_bad_pixels_info(bad_pixels, path):
    "[номер строки];[номер столбца];[номер канала];[«битое» значение];[исправленноезначение]"
    pd.DataFrame(bad_pixels).to_csv(path, sep=";", index=False, header=False)

def CreateGeoTiff(outRaster, data, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, no_bands  = data.shape # c,h,w
    DataSet = driver.Create(outRaster, cols, rows, no_bands, gdal.GDT_UInt16)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)
    data = np.moveaxis(data, -1, 0)

    for i, image in enumerate(data, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    DataSet = None



def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--crop_name", type=str, default="./18.Sitronics/1_20/crop_0_0_0000.tif")
    parser.add_argument("--save_dir", type=str, default="./data/restored_images")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    os.makedirs(args.save_dir, exist_ok=True)

    img_geo = gdal.Open(args.crop_name)
    img = img_geo.ReadAsArray()

    restored_img, bad_pixels = restore_pixels_4channel(img, restore_by_mean)
    path_to_save_img = os.path.join(args.save_dir, os.path.basename(args.crop_name))
    CreateGeoTiff(path_to_save_img, restored_img, img_geo.GetGeoTransform(), img_geo.GetProjection())
    path_to_save_csv = path_to_save_img.rsplit(".", 1)[0] + ".csv"
    save_bad_pixels_info(bad_pixels, path_to_save_csv)