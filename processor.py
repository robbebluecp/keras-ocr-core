import keras.backend as K
import config
from models import model
import numpy as np
from tools import utils_image, utils
from PIL import Image

ss = K.get_session()


def get_yolo_output(image_data, wh, new_wh):
    w, h = wh
    new_w, new_h = new_wh
    darknet = model.DarkNet()(n_class=len(config.class_names), n_anchor=len(config.anchors))
    darknet.load_weights(config.yolo3_weights_path)
    yolo_output = darknet.predict(image_data)
    boxes = []
    scores = []
    for l in range(len(yolo_output)):
        box_xy, box_wh, box_confidence, box_class_probs = model.yolo_core(yolo_output[l], config.anchors[config.anchor_mask[l]], len(config.class_names), [new_h, new_w])
        box_score = box_confidence * box_class_probs
        box_score = K.reshape(box_score, [-1, len(config.class_names)])
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)
        # xmin, ymin, xmax, ymax    (1, None, None, 3, 4)
        box = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        box = K.reshape(box, [-1, 4])
        # (N, 4)
        boxes.append(box)
        # (N, 2)
        scores.append(box_score)

    # (NN, 2)
    boxes = K.concatenate(boxes, axis=0)
    scores = K.concatenate(scores, axis=0)

    boxes, scores = ss.run(boxes), ss.run(scores)
    scores = scores[..., 1]
    boxes *= np.asarray([w, h, w, h])

    boxes[:, 0:4][boxes[:, 0:4] < 0] = 0
    boxes[:, 0][boxes[:, 0] >= w] = w - 1
    boxes[:, 1][boxes[:, 1] >= h] = h - 1
    boxes[:, 2][boxes[:, 2] >= w] = w - 1
    boxes[:, 3][boxes[:, 3] >= h] = h - 1

    return boxes, scores


def correct_boxes(boxes, scores, shape, img):
    scores = scores[:, np.newaxis]


    # nms for text proposals
    if not len(boxes):
        return []
        # (N, 4), (N, )
    # NMS
    boxes, scores = utils_image.cv2_nms(boxes, scores, config.score_threshold, config.nms_threshold)
    # scores = normalize(scores)
    if scores.shape[0] == 0:
        return []
    max_ = scores.max()
    min_ = scores.min()
    scores = (scores - min_) / (max_ - min_) if max_ - min_ != 0 else scores - min_
    # (x0, y0, x1, y1, scores, xx, yy, height+2.5)=(N, 8), (N, )
    text_lines, scores = utils.get_box_lines(boxes, scores, shape)
    text_lines = utils.get_bboxes(text_lines)
    text_lines, scores = utils_image.cv2_nms_rotate(text_lines, scores, config.score_threshold_rotated, config.nms_threshold_rotated)
    im = Image.fromarray(img)
    newBoxes = []
    for index, box in enumerate(text_lines):
        partImg, box = utils_image.rotate_cut_img(im, box, config.left, config.right)
        box['img'] = partImg.convert('L')
        newBoxes.append(box)
    return newBoxes


def get_crnn_output(boxes):
    crnn = model.CRNN()()
    crnn.load_weights(config.crnn_weights_path)
    for i in range(len(boxes)):
        image = boxes[i]['img']
        image = utils_image.resizeNormalize(image, 32)
        image = image.astype(np.float32)
        image = np.array([[image]])
        preds = crnn.predict(image)
        preds = np.argmax(preds, axis=-1).reshape((-1,))
        boxes[i]['text'] = utils.strLabelConverter(preds, config.alphabet)
    return boxes
