import numpy as np
from nms import rotate_nms
from image import get_boxes
from tools import utils_image


def detect(text_proposals, scores, shape):
    scores = scores[:, np.newaxis]
    # 置信度
    TEXT_PROPOSALS_MIN_SCORE = 0.1
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    TEXT_LINE_NMS_THRESH = 0.99  ##文本行之间测iou值
    LINE_MIN_SCORE = 0.1

    # nms for text proposals
    if not len(text_proposals):
        return []
        # (N, 4), (N, )
    # NMS
    text_proposals, scores = utils_image.cv2_nms(text_proposals, scores, TEXT_PROPOSALS_MIN_SCORE, TEXT_PROPOSALS_NMS_THRESH)
    # scores = normalize(scores)
    if scores.shape[0] == 0:
        return []
    max_ = scores.max()
    min_ = scores.min()
    scores = (scores - min_) / (max_ - min_) if max_ - min_ != 0 else scores - min_
    # (x0, y0, x1, y1, scores, xx, yy, height+2.5)=(N, 8), (N, )
    text_lines, scores = get_text_lines(text_proposals, scores, shape)
    text_lines = get_boxes(text_lines)
    text_lines, scores = rotate_nms(text_lines, scores, LINE_MIN_SCORE, TEXT_LINE_NMS_THRESH)

    return text_lines, scores


def get_text_lines(text_proposals, scores, shape):
    # (N, M)
    tp_groups = build_graph(text_proposals, scores, shape, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6)
    # (N, 8)
    text_lines = np.zeros((len(tp_groups), 8), np.float32)
    # (N, )
    newscores = np.zeros((len(tp_groups),), np.float32)
    for index, tp_indices in enumerate(tp_groups):
        # (NN, 4) type: list
        text_line_boxes = text_proposals[list(tp_indices)]
        # num = np.size(text_line_boxes)##find
        # 中心点
        # (N, )
        X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
        # (N, )
        Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

        x0 = np.min(text_line_boxes[:, 0])
        x1 = np.max(text_line_boxes[:, 2])

        offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

        # 拟合上线
        lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
        # 拟合下线
        lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

        # the score of a text line is the average score of the scores
        # of all text proposals contained in the text line
        score = scores[list(tp_indices)].sum() / float(len(tp_indices))

        # return (K, b)
        # p1 = np.poly1d(z1)
        z1 = np.polyfit(X, Y, 1)
        text_lines[index, 0] = x0
        text_lines[index, 1] = min(lt_y, rt_y)
        text_lines[index, 2] = x1
        text_lines[index, 3] = max(lb_y, rb_y)
        text_lines[index, 4] = score
        text_lines[index, 5] = z1[0]
        text_lines[index, 6] = z1[1]
        height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))
        text_lines[index, 7] = height + 2.5
        newscores[index] = score

    # (x0, y0, x1, y1, scores, xx, yy, height+2.5)=(N, 8), (N, )
    return text_lines, newscores


def fit_y(X, Y, x1, x2):
    """

    :param X:       x0
    :param Y:       x1
    :param x1:      x0+xx
    :param x2:
    :return:
    """
    # if X only include one point, the function will get line y=Y[0]
    if np.sum(X == X[0]) == len(X):
        return Y[0], Y[0]
    # 函数mapping
    p = np.poly1d(np.polyfit(X, Y, 1))
    return p(x1), p(x2)


def build_graph(text_proposals, scores, shape, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
    heights = text_proposals[:, 3] - text_proposals[:, 1]

    # [w, ??]
    boxes_table = [[] for _ in range(shape[1])]
    for index, box in enumerate(text_proposals):
        # print(int(box[0]),len(boxes_table))
        boxes_table[int(box[0])].append(index)
    print(boxes_table)

    graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

    for index, box in enumerate(text_proposals):

        successions = []
        # 扫描, 横向
        for left in range(int(box[0]) + 1, min(int(box[0]) + MAX_HORIZONTAL_GAP + 1, shape[1])):
            adj_box_indices = boxes_table[left]
            for adj_box_index in adj_box_indices:
                if meet_v_iou(adj_box_index, index, heights, text_proposals, MIN_V_OVERLAPS, MIN_SIZE_SIM):
                    successions.append(adj_box_index)
            # 找到一个连续区域就撤
            if len(successions) != 0:
                break
        # successions = self.get_successions(index)
        if len(successions) == 0:
            continue

        succession_index = successions[np.argmax(scores[successions])]

        # precursors = self.get_precursors(succession_index)
        precursors = []
        box = text_proposals[succession_index]
        for left in range(int(box[0]) - 1, max(int(box[0] - MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = boxes_table[left]
            for adj_box_index in adj_box_indices:
                if meet_v_iou(adj_box_index, succession_index, heights, text_proposals, MIN_V_OVERLAPS, MIN_SIZE_SIM):
                    precursors.append(adj_box_index)
            if len(precursors) != 0:
                break

        if scores[index] >= np.max(scores[precursors]):
            flag = True
        else:
            flag = False

        if flag:
            # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
            # have equal scores.
            graph[index, succession_index] = True
    # return self.graph_(graph)
    sub_graphs = []
    for index in range(graph.shape[0]):
        # 理论上相互独立，很独特的逻辑
        if not graph[:, index].any() and graph[index, :].any():
            v = index
            sub_graphs.append([v])
            while graph[v, :].any():
                v = np.where(graph[v, :])[0][0]
                sub_graphs[-1].append(v)
    # print(sub_graphs)
    return sub_graphs


def meet_v_iou(index1, index2, heights, text_proposals, MIN_V_OVERLAPS, MIN_SIZE_SIM):
    # 竖直方向重叠占比, 比例超过0.3
    def overlaps_v(index1, index2):
        h1 = heights[index1]
        h2 = heights[index2]
        y0 = max(text_proposals[index2][1], text_proposals[index1][1])
        y1 = min(text_proposals[index2][3], text_proposals[index1][3])
        return max(0, y1 - y0) / min(h1, h2)

    # size比较，相似度差超过0.6
    def size_similarity(index1, index2):
        h1 = heights[index1]
        h2 = heights[index2]
        return min(h1, h2) / max(h1, h2)

    return overlaps_v(index1, index2) >= MIN_V_OVERLAPS and size_similarity(index1, index2) >= MIN_SIZE_SIM
