# example of loading a pix2pix model and using it for one-off image translation
import osr
import gdal
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import geopandas as gp
from pathlib import Path
import os
from PIL import Image
from skimage.morphology import skeletonize
from skimage.io import imread
from skimage.io import imsave
from scipy.spatial import Voronoi, voronoi_plot_2d

# load an image
def load_image(myfile, size=(256,256)):
    # load image with the preferred size
    pixels = load_img(myfile, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1] las gans trabajan centradas en el origen
    #pixels = (pixels - 127.5) / 127.5
    pixels = pixels / 255.
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

raster_in_folder_name = 'raster_in'
data_folder_name = 'raster_to_shape'
shape_out_folder_name = 'shape_out'
raster_out_folder_name = 'raster_out'

segmenter_path = 'model'
data_tiles_path = Path(data_folder_name + '/tiles')
data_masks_path = Path(data_folder_name + '/masks')

tile_width = 256
tile_height = 256

modelname = 'model_161300.h5'
#modelname = 'model_035340_plain_roads.h5'
#modelname = 'model_007300_rivers.h5'
#modelname = 'model_007200_green.h5'

type = modelname

model = load_model(segmenter_path+'/'+modelname)

for rasterin in os.listdir(raster_in_folder_name):
    os.system('venv/bin/gdal_retile.py -ps ' + str(tile_width) + ' ' + str(tile_height) + ' -targetDir ' + data_folder_name + '/tiles ' + raster_in_folder_name + '/' + rasterin)
    path, dirs, files = next(os.walk(data_folder_name + '/tiles'))
    i = 0
    for img_name in os.listdir(data_folder_name + '/tiles'):
        i = i + 1
        print(str(i)+" / "+str(len(files)), end="\r")
        mask_name = 'mask_' + img_name
        jpg_img_name = img_name.replace('.tif','.jpg')
        jpg_mask_name = mask_name.replace('.tif','.jpg')
        
        source_raster = gdal.Open(data_folder_name + '/tiles/' + img_name)
        srs_geotrans = source_raster.GetGeoTransform()
        srs_proj = source_raster.GetProjection()

        raster_path = data_tiles_path/img_name
        img = load_image(raster_path)
        res_img = model.predict(img)
        res_img = (res_img + 1) / 2.0

        mpimg.imsave(data_masks_path/jpg_img_name, res_img[0]/255)
        mpimg.imsave(data_masks_path/jpg_mask_name, res_img[0]/255)

        rs_res_img = Image.open(data_folder_name + '/masks/' + jpg_img_name)
        rs_res_img = rs_res_img.resize((tile_width, tile_height))
        rs_res_img.save(data_folder_name + '/masks/' + img_name)

        rs_res_mask = Image.open(data_folder_name + '/masks/' + jpg_mask_name)
        rs_res_mask = rs_res_mask.resize((tile_width, tile_height))
        #rs_res_mask = rs_res_mask.convert('1',colors=1)
        rs_res_mask.save(data_folder_name + '/masks/' + mask_name)

        dataset = gdal.Open(data_folder_name + '/masks/' + mask_name)

        dataset = gdal.Open(data_folder_name + '/masks/' + mask_name)

        width = dataset.RasterXSize
        height = dataset.RasterYSize
        datas = dataset.ReadAsArray(0,0,width,height)
        driver = gdal.GetDriverByName("GTiff")
        tods = driver.Create(data_folder_name + '/masks/srs_' + img_name,width,height,3,options=["INTERLEAVE=PIXEL"])
        tods.SetProjection(srs_proj)
        tods.SetGeoTransform(srs_geotrans)
        tods.WriteRaster(0,0,width,height,datas.tostring(),width,height,band_list=[1])
        tods = driver.Create(data_folder_name + '/masks/srs_' + img_name,width,height,3,options=["INTERLEAVE=PIXEL"])
        tods.SetProjection(srs_proj)
        tods.SetGeoTransform(srs_geotrans)
        tods.WriteRaster(0,0,width,height,datas.tostring(),width,height,band_list=[1])

    inRasfn = data_folder_name + '/masks/srs_'
    outRasfn = raster_out_folder_name + '/' + rasterin.split('.')[0] + '_' + type
    
    outSHPfn = shape_out_folder_name + '/' + rasterin.split('.')[0] + '_' + type
    os.system('venv/bin/gdal_merge.py -o ' + outRasfn + '.tif' + ' ' + inRasfn + '*.tif')
    if(type == 'model_161300.h5x'):
        outSHPfn = outSHPfn + '_sk'
        source_raster = gdal.Open(outRasfn + '.tif')
        srs_geotrans = source_raster.GetGeoTransform()
        srs_proj = source_raster.GetProjection()

        image_to_sk = imread(outRasfn + '.tif')
        rs_res_mask = skeletonize(image_to_sk)
        imsave(outRasfn + '.tif',rs_res_mask)
        source_raster = gdal.Open(outRasfn + '.tif')
        width = source_raster.RasterXSize
        height = source_raster.RasterYSize
        datas = source_raster.ReadAsArray(0,0,width,height)
        driver = gdal.GetDriverByName("GTiff")
        tods = driver.Create(outRasfn + '.tif',width,height,3,options=["INTERLEAVE=PIXEL"])
        tods.SetProjection(srs_proj)
        tods.SetGeoTransform(srs_geotrans)
        tods.WriteRaster(0,0,width,height,datas.tostring(),width,height,band_list=[1])
        tods = driver.Create(outRasfn + '.tif',width,height,3,options=["INTERLEAVE=PIXEL"])
        tods.SetProjection(srs_proj)
        tods.SetGeoTransform(srs_geotrans)
        tods.WriteRaster(0,0,width,height,datas.tostring(),width,height,band_list=[1])

    if os.path.exists(outSHPfn + '.shp'):
        os.remove(outSHPfn + '.shp')
    if os.path.exists(outSHPfn + '.dbf'):
        os.remove(outSHPfn + '.dbf')
    if os.path.exists(outSHPfn + '.prj'):
        os.remove(outSHPfn + '.prj')
    if os.path.exists(outSHPfn + '.cpg'):
        os.remove(outSHPfn + '.cpg')
    if os.path.exists(outSHPfn + '.shx'):
        os.remove(outSHPfn + '.shx')
    
    for file in os.listdir(data_folder_name + '/tiles'):
        os.remove(data_folder_name + '/tiles/' + file)

    for file in os.listdir(data_folder_name + '/shapes'):
        os.remove(data_folder_name + '/shapes/' + file)

    for file in os.listdir(data_folder_name + '/masks'):
        os.remove(data_folder_name + '/masks/' + file)

    os.system('venv/bin/gdal_polygonize.py -8 -f "ESRI Shapefile" -mask ' + outRasfn + '.tif' + ' ' + outRasfn + '.tif' + ' ' + outSHPfn + '.shp')
    gdf = gp.read_file(outSHPfn + '.shp')
    #gdf.buffer(5000,resolution=100)
    buf = gp.read_file(outSHPfn + '.shp')
    sim = gp.read_file(outSHPfn + '.shp')
    #buf['geometry'] = Voronoi(buf['geometry'])
    buf.to_file(outSHPfn + '_buf.shp')
    tolerance = 5
    sim = sim.simplify(tolerance, preserve_topology=True)
    sim.to_file(outSHPfn + '_sim.shp')
    gpproj = gdf.crs
    gdf['x'] = gdf.centroid.map(lambda p: p.x)
    gdf['y'] = gdf.centroid.map(lambda p: p.y)
    gdf['type']= type
    del gdf['DN']
    gdf.insert(0, 'ID', range(1, 1 + len(gdf)))
    gdf = gdf.to_crs('EPSG:3116')
    gdf['area']= gdf['geometry'].area
    gdf = gdf.to_crs(gpproj)
    gdf.to_file(outSHPfn + '.shp')
    
