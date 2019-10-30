import cv2
from tools import utils_image
import numpy as np
import processor


# 图片预处理
img = cv2.imread('tmp3.png')
h, w, _ = shape = img.shape
new_w, new_h = utils_image.scale_image(w, h)
print(new_w, new_h)
image_data = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
image_data = np.array(image_data, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)


# 提议框检测部分
boxes, scores = processor.get_yolo_output(image_data, (w, h), (new_w, new_h))


# 检测区域提取
new_boxes = processor.correct_boxes(boxes, scores, shape, img)
# for i in new_boxes:
#     p1 = (int(i['cx'] - 0.5 * i['w']), int(i['cy'] - 0.5 * i['h']))
#     p2 = (int(i['cx'] + 0.5 * i['w']), int(i['cy'] + 0.5 * i['h']))
#     cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
#
# cv2.imshow('', img)

items = processor.get_crnn_output(new_boxes)
print(items)

# cv2.waitKey()
# cv2.destroyAllWindows()
