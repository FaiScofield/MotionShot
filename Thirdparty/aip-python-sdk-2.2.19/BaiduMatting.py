#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import base64
import numpy as np
import os
import shutil

from aip import AipBodyAnalysis


def mattingApi(file_name, client):
    ori_img = cv2.imread(file_name)
    height, width, _ = ori_img.shape

    with open(file_name, 'rb') as fp:
        image = fp.read()

    seg_res = client.bodySeg(image)
    labelmap = base64.b64decode(seg_res['scoremap'])  # labelmap, scoremap
    nparr = np.fromstring(labelmap, np.uint8)
    labelimg = cv2.imdecode(nparr, 1)
    labelimg = cv2.resize(labelimg, (width, height),
                          interpolation=cv2.INTER_NEAREST)
#    labelimg        = labelimg[:,:,0]

#    foreground = base64.b64decode(seg_res['foreground'])
#    nparr_foreground = np.fromstring(foreground, np.uint8)
#    foregroundimg = cv2.imdecode(nparr_foreground, 1)
#    foregroundimg = cv2.resize(foregroundimg, (width, height), interpolation=cv2.INTER_NEAREST)
#    im_foreground = np.where(foregroundimg==1, 10, foregroundimg)

#    mask = np.where(labelimg==1, 255, labelimg)
    foreground_img = cv2.bitwise_and(ori_img, labelimg)

    return labelimg, foreground_img
#    return labelimg


# client config
APP_ID = '18679385'
API_KEY = '12yptwaZPOxoGBfPR0PGYT43'
SECRET_KEY = 'sx8w8l1dzlD2Gt0QAKgZxItRB3uE8DZz'
client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)


src_dir = '/home/vance/dataset/rk/Phone/withGT3/demo3/image_rect'
#output_dir = '/home/vance/dataset/rk/Phone/withGT2/girl-walk-sj-7/gt_rect_baidu'
images = os.listdir(src_dir)

#img_index = 1
for file_name in images:
    if '.png' in file_name and '_image' not in file_name:
#    if '.jpg' in file_name and '_image' not in file_name:

        file_name = os.path.join(src_dir, file_name)
        print("dealing with " + file_name)
#        mask        = mattingApi(file_name, client)
        mask, foreground = mattingApi(file_name, client)

#        image_name = output_dir + '_image.jpg'
#        label_name = output_dir + '/%d.png' % img_index
#        fore_name = output_dir + '/%d_fore.jpg' % img_index
#        img_index = img_index + 1

        label_name = file_name + '_mask.png'
        fore_name = file_name + '_fore.jpg'

#        shutil.move(os.path.join(src_dir, file_name), os.path.join(src_dir, image_name))
        cv2.imwrite(os.path.join(src_dir, label_name), mask)
        cv2.imwrite(os.path.join(src_dir, fore_name), foreground)

print('done.')
