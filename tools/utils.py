import numpy as np


def get_box_lines(text_proposals, scores, shape):
    """
    拟合生成框边界
    :param text_proposals:
    :param scores:
    :param shape:
    :return:
    """
    def fit_y(X, Y, x1, x2):
        """
        拟合直线
        :param X:
        :param Y:
        :param x1:
        :param x2:
        :return:
        """
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        # 函数mapping
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    # (N, M)
    connected_regions = get_connected_region(text_proposals, scores, shape, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6)
    # (N, 8)
    text_lines = np.zeros((len(connected_regions), 8), np.float32)
    # (N, )
    new_scores = np.zeros((len(connected_regions),), np.float32)
    for index, tp_indices in enumerate(connected_regions):
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
        new_scores[index] = score

    # (x0, y0, x1, y1, scores, xx, yy, height+2.5)=(N, 8), (N, )
    return text_lines, new_scores


def get_connected_region(text_proposals, scores, shape, MAX_HORIZONTAL_GAP=30, MIN_V_OVERLAPS=0.6, MIN_SIZE_SIM=0.6):
    """
    生成连结区域列表
    :param text_proposals:
    :param scores:
    :param shape:
    :param MAX_HORIZONTAL_GAP:
    :param MIN_V_OVERLAPS:
    :param MIN_SIZE_SIM:
    :return:
    """

    def meet_v_iou(index1, index2, heights, text_proposals, MIN_V_OVERLAPS, MIN_SIZE_SIM):
        """
        判断区域是否连结
        :param index1:
        :param index2:
        :param heights:
        :param text_proposals:
        :param MIN_V_OVERLAPS:
        :param MIN_SIZE_SIM:
        :return:
        """
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
    heights = text_proposals[:, 3] - text_proposals[:, 1]

    # [w, ??]
    boxes_table = [[] for _ in range(shape[1])]
    for index, box in enumerate(text_proposals):
        # print(int(box[0]),len(boxes_table))
        boxes_table[int(box[0])].append(index)

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


def get_bboxes(bboxes):
    """
    生成边界框
    :param bboxes:
    :return:
    """
    # (x0, y0, x1, y1, scores, xx, yy, height+2.5)=(N, 8), (N, )
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    boxes = []
    for box in text_recs:
        x1, y1 = (box[0], box[1])
        x2, y2 = (box[2], box[3])
        x3, y3 = (box[6], box[7])
        x4, y4 = (box[4], box[5])
        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    boxes = np.array(boxes)

    return boxes


def strLabelConverter(res, alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)