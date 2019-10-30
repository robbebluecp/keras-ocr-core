# -*- coding: utf-8 -*-

import numpy as np

alphabet_file = 'data.txt'
with open(alphabet_file, 'r') as f:
    alphabet = f.read().strip()
    f.close()

anchors = np.asarray([[8,11], [8,16], [8,23], [8,33], [8,48], [8,97], [8,139], [8,198], [8,283]])
anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
class_names = ['none', 'text', ]

yolo3_weights_path = 'model_data/yolo_ocr.h5'
crnn_weights_path = 'model_data/ocr.h5'

score_threshold = 0.1
nms_threshold = 0.3
score_threshold_rotated = 0.1
nms_threshold_rotated = 0.99
left = 0.1
right = 0.1