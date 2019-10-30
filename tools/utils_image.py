import numpy as np
from PIL import Image
import cv2


def scale_image(w, h, min_scale=600, max_scale=900):
    """
    调整图片尺寸，放缩至短边<=600，长边<=900范围内
    :param w:
    :param h:
    :param min_scale:
    :param max_scale:
    :return:
    """
    ratio = float(min_scale) / min(h, w)
    if ratio * max(h, w) > max_scale:
        ratio = float(max_scale) / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)

    return new_w - (new_w % 32), new_h - (new_h % 32)


def cv2_nms(boxes, scores, score_threshold, nms_threshold):
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0:2] = boxes[:, 0:2]
    new_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    new_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    new_boxes, new_scores = new_boxes.tolist(), scores.tolist()
    new_scores = list(map(lambda x: x[0], new_scores))
    index = cv2.dnn.NMSBoxes(new_boxes, new_scores, score_threshold, nms_threshold)
    index = index.reshape((-1,))
    return boxes[index], scores[index]


def cv2_nms_rotate(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    new_boxes = []
    cx = (boxes[:, 0] + boxes[:, 2] + boxes[:, 6] + boxes[:, 6]) / 4.0
    cy = (boxes[:, 1] + boxes[:, 3] + boxes[:, 5] + boxes[:, 7]) / 4.0
    w = (np.sqrt((boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 1]) ** 2) + np.sqrt((boxes[:, 4] - boxes[:, 6]) ** 2 + (boxes[:, 5] - boxes[:, 7]) ** 2)) / 2
    h = (np.sqrt((boxes[:, 2] - boxes[:, 4]) ** 2 + (boxes[:, 3] - boxes[:, 5]) ** 2) + np.sqrt((boxes[:, 1] - boxes[:, 6]) ** 2 + (boxes[:, 1] - boxes[:, 7]) ** 2)) / 2
    sinA = (h * (boxes[:, 0] - cx) - w * (boxes[:, 1] - cy)) * 1.0 / (h * h + w * w) * 2
    for i in range(len(sinA)):
        if abs(sinA[i]) > 1:
            angel = 0.0
        else:
            angel = np.arcsin(sinA[i])
        new_boxes.append(((cx[i], cy[i]), (w[i], h[i]), angel))

    new_scores = scores.tolist()
    index = cv2.dnn.NMSBoxesRotated(new_boxes, new_scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index) > 0:
        index = index.reshape((-1,))
        return boxes[index], scores[index]
    else:
        return [], []


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy

def rotate_cut_img(im, box, left, right):
    angle, w, h, cx, cy = solve(box)
    degree_ = angle * 180.0 / np.pi

    box = (max(1, cx - w / 2 - left * (w / 2))  ##xmin
           , cy - h / 2,  ##ymin
           min(cx + w / 2 + right * (w / 2), im.size[0] - 1)  ##xmax
           , cy + h / 2)  ##ymax
    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(cx, cy)).crop(box)
    box = {'cx': cx, 'cy': cy, 'w': newW, 'h': newH, 'degree': degree_, }
    return tmpImg, box



def resizeNormalize(img, imgH=32):
    scale = img.size[1] * 1.0 / imgH
    w = img.size[0] / scale
    w = int(w)
    img = img.resize((w, imgH), Image.BILINEAR)
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    return img





if __name__ == '__main__':
    import cv2
    from PIL import Image

    img = cv2.imread('../tmp.png')
    w, h, _ = img.shape

    img2 = Image.fromarray(img)
    ww, hh = img2.size

    new_ww, new_hh = scale_image(ww, hh)

    img = cv2.resize(img, (new_ww, new_hh), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img2 = img2.resize((new_ww, new_hh), Image.BICUBIC)
    img2.show()
