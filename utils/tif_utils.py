import matplotlib.pyplot as plt

import gdal


def read_tif_rgbir(tif_file):
    dataset = gdal.Open(tif_file)
    r = dataset.GetRasterBand(1).ReadAsArray()
    g = dataset.GetRasterBand(2).ReadAsArray()
    b = dataset.GetRasterBand(3).ReadAsArray()
    ir = dataset.GetRasterBand(4).ReadAsArray()

    return r, g, b, ir

def show_tif(tif_file):
    dataset = gdal.Open(tif_file)
    array = dataset.GetRasterBand(1).ReadAsArray()

    plt.figure(figsize=(10, 10))
    plt.imshow(array)
    plt.colorbar()