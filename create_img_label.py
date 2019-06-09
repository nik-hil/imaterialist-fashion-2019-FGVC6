import numpy as np
import pandas as pd

from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

from itertools import groupby
from progressbar import ProgressBar
import cv2
from multiprocessing import Pool
import os
import json
import torchvision
from datetime import datetime

category_num = 46 + 1

class ImageMask():
    masks = {}
    
    def make_mask_img(self, segment_df):
        seg_width = segment_df.iloc[0].Width
        seg_height = segment_df.iloc[0].Height
        
        seg_img = np.copy(self.masks.get((seg_width, seg_height)))
        try:
            if not seg_img:
                seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
                self.masks[(seg_width, seg_height)] = np.copy(seg_img)
        except:
            # seg_img exists
            pass
        for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
            pixel_list = list(map(int, encoded_pixels.split(" ")))
            for i in range(0, len(pixel_list), 2):
                start_index = pixel_list[i] - 1
                index_len = pixel_list[i+1] - 1
                if int(class_id.split("_")[0]) < category_num - 1:
                    seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
        seg_img = seg_img.reshape((seg_height, seg_width), order='F')
        return seg_img
       

img_mask = ImageMask()
def create_label(img):
    fname = path_lbl.joinpath(os.path.splitext(img)[0] + '_P.png').as_posix()
    if os.path.isfile(fname): # skip
        return
    img_df = df[df.ImageId == img]
    mask = img_mask.make_mask_img(img_df)
    img_mask_3_chn = np.dstack((mask, mask, mask))
    cv2.imwrite(fname, img_mask_3_chn)
    print(".", end="")


path = Path('/home/jupyter/comp/')
path_lbl = path/'labels'
path_img = path/'train'
df = pd.read_csv(path/'train.csv')
images = df.ImageId.unique()


# Keep eye here and run it again if processing stops
with Pool(processes=5) as pool:
    pool.map(create_label, images)
