#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 21:44:44 2022

@author: benjaminglass
"""

import ee
import matplotlib.pyplot as plt
import earthpy as ep
import rioxarray as rxr
import geemap as gm
import numpy as np
import pandas as pd
# ee.Authenticate()
# ee.Initialize()


#// Define a Geometry object.
MtAbe_geom = ee.Geometry({
  'type': 'Polygon',
  'coordinates':
    [[[-72.95921456312732,44.10166265571179],
      [-72.90702950453357,44.13493647921331],
      [-72.95921456312732,44.13493647921331],
      [-72.95921456312732,44.10166265571179]]]
},None,False);
    
SRTM = ee.Image("USGS/SRTMGL1_003")

# SRTM_sr = SRTM.sampleRectangle(region=MtAbe_geom).get('elevation').getInfo()
# SRTM_arr = np.array(SRTM_sr)
# print(type(SRTM_arr))
# print(SRTM_arr.shape)


# //specify start and end dates
# //var start = '2013-03-18';
start = '2013-07-31'
end = '2021-08-31'

# //load and filter landsat 8 imagery
#ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).sort('CLOUD_COVER')
#print(ls8.first())


#GET IMAGES BY INDIVIDUAL NAME
ls8 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013029_20141003')

# ls8_sr = ls8.first().sampleRectangle(region=MtAbe_geom)

# //Map.addLayer(ls8);

# ls8_ndvi = ls8.map(function(image) {
#   var nir = image.select('SR_B5');
#   var red = image.select('SR_B4');
#   var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
#   var date = image.get('DATE_ACQUIRED');
#   //clip ndvi image by geometry
#   var ndvi_clip = ndvi.clip(MtAbe_geom);
  
#   //save average to image as new feature
#   var new_im = ee.Image(ndvi_clip)
                              
#   return ee.Image(new_im);
# });

def normalized_diff(b1,b2):
    return ((b1-b2)/(b1+b2))

nir = ls8.select('SR_B5')
red = ls8.select('SR_B4')

ndvi_test = nir.subtract(red).divide(nir.add(red)).rename('NDVI')



# ndvi_test_sr = ls8.sampleRectangle(region=MtAbe_geom)
# ndvi_arr_b4 = np.array(ndvi_test_sr.get('SR_B4').getInfo())
# ndvi_arr_b5 = np.array(ndvi_test_sr.get('SR_B5').getInfo())

# print(ndvi_arr_b4)
# print(ndvi_arr_b5)

# print(ndvi_arr_b4.shape)

#add two arrays together




#print(ls8_test)

# fig, ax = plt.subplots(figsize=(14, 6))
# ax.plot(ls8_test)
# plt.show()



#minMaxValues = ndvi_test.reduceRegion({reducer: ee.Reducer.minMax()});

#print(ndvi_test.min());

#print(ndvi_test)





# Map.addLayer(ls8_ndvi.sort('CLOUD_COVER').first(),{palette: ['white','green']});


# var ind_ev = SRTM.divide(ls8_ndvi.sort('CLOUD_COVER').first());

# print(ind_ev);
# Map.addLayer(ind_ev,{min: 1225, max: 7142, palette: ['white','red']});

# //var minMaxValues = ind_ev.reduceRegion({reducer: ee.Reducer.minMax()});

# print(minMaxValues);


min_elev = 1300
max_elev = 1400
SRTM_mask = SRTM.updateMask(SRTM.gt(min_elev))
SRTM_mask = SRTM_mask.updateMask(SRTM_mask.lte(max_elev)).clip(MtAbe_geom)
ind_ev = ndvi_test.clip(MtAbe_geom).mask(SRTM_mask)
mean_ndvi = ind_ev.reduceRegion(reducer=ee.Reducer.mean())

#print(np.array(mean_ndvi.get('NDVI').getInfo()))



max_value = 1451
step = 5

test = np.array(list(range(1, max_value,step))).T
data = pd.DataFrame(test,columns=['elev_steps'])
print(data)


def add_ndvi_image(image,data):
    
    temp_arr = []
    for index,row in data.iterrows():
        #print(index,row)
        min_elev = row['elev_steps']
        max_elev = min_elev+(step-1)
        print(min_elev)
        
        timestamp = image.get('system:index').getInfo()
        print(timestamp)
        
        data[timestamp] = 'default'
        
        nir = image.select('SR_B5')
        red = image.select('SR_B4')

        ndvi_test = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        
        SRTM_mask = SRTM.updateMask(SRTM.gt(min_elev))
        SRTM_mask = SRTM_mask.updateMask(SRTM_mask.lte(max_elev)).clip(MtAbe_geom)
        ind_ev = ndvi_test.clip(MtAbe_geom).mask(SRTM_mask)
        
        mean_ndvi = ind_ev.reduceRegion(reducer=ee.Reducer.mean())
        
        #print(type(mean_ndvi.get('NDVI').getInfo()))
        
        mean_ndvi = np.array(mean_ndvi.get('NDVI').getInfo())
        #print(mean_ndvi)
        #print(type(mean_ndvi.item(0)))
        #print(mean_ndvi.item(0))
        
        temp_arr.append(mean_ndvi.item(0))
        
        #temp_arr.append(1)
        
        #print(temp_arr)
        #data.loc[[index]][timestamp] = mean_ndvi
        #data.at[index,timestamp] = mean_ndvi.item(0)
        
        #print(data)
    
    # print(temp_arr)
    # print(data)
    
    data[timestamp] = temp_arr
    data = data[['elev_steps',timestamp]]
    
    return(data)


#LS8 IMAGES FROM 5/31-8/31 2013
#-------------------------------
# 0: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130820 (19 bands)
# 1: Image LANDSAT/LC08/C02/T1_L2/LC08_013029_20130712 (19 bands)
# 2: Image LANDSAT/LC08/C02/T1_L2/LC08_013030_20130712 (19 bands)
# 3: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130601 (19 bands)
# 4: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130719 (19 bands)
# 5: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130804 (19 bands)
# 6: Image LANDSAT/LC08/C02/T1_L2/LC08_013029_20130610 (19 bands)
# 7: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130703 (19 bands)
# 8: Image LANDSAT/LC08/C02/T1_L2/LC08_014029_20130617 (19 bands)
# 9: Image LANDSAT/LC08/C02/T1_L2/LC08_013030_20130728 (19 bands)
# 10: Image LANDSAT/LC08/C02/T1_L2/LC08_013029_20130728 (19 bands)
# 11: Image LANDSAT/LC08/C02/T1_L2/LC08_013030_20130610 (19 bands)


ls8_1 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013029_20130712')
ls8_2 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013030_20130712')
ls8_3 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_014029_20130601')
ls8_4 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_014029_20130719')
ls8_5 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_014029_20130804')
ls8_6 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013029_20130610')
ls8_7 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_014029_20130703')
ls8_8 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_014029_20130617')
ls8_9 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013030_20130728')
ls8_10 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013029_20130728')
ls8_11 = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_013030_20130610')

data_1 = add_ndvi_image(ls8_1,data)
data_2 = add_ndvi_image(ls8_2,data)
data_3 = add_ndvi_image(ls8_3,data)
data_4 = add_ndvi_image(ls8_4,data)
data_5 = add_ndvi_image(ls8_5,data)
data_6 = add_ndvi_image(ls8_6,data)
data_7 = add_ndvi_image(ls8_7,data)
data_8 = add_ndvi_image(ls8_8,data)
data_9 = add_ndvi_image(ls8_9,data)
data_10 = add_ndvi_image(ls8_10,data)
data_11 = add_ndvi_image(ls8_11,data)

data_df = data_1.merge(data_2,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_3,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_4,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_5,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_6,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_7,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_8,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_9,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_10,left_on='elev_steps',right_on='elev_steps')
data_df = data_df.merge(data_11,left_on='elev_steps',right_on='elev_steps')

data_df

data_df.columns

data_df.head()

print(data_df.iloc[:,1:11])


data_df['mean'] = data_df.iloc[:, 1:11].mean(axis=1)

data_df['mean']


#new_df = pd.DataFrame(new_df,columns=['min_elev','avg_ndvi'])
#print(new_df)


ls8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterBounds(MtAbe_geom) \
    .filterDate(start, end) \
    .sort('CLOUD_COVER')

def get_ids(image):
    return image.get('system:index')

ls8_imgs = ls8.map(get_ids)




title = 'Average NDVI values from May-Aug 2013 for Landsat 8 pixels \n around Mt Abraham (Lincoln, VT)'

f, ax = plt.subplots()
plt.scatter(data_df['elev_steps'], data_df['mean'])
plt.title(title)
plt.xlabel("Elevation band (every 10m)")
plt.ylabel("NDVI value")
plt.show()

    
    
    