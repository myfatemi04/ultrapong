import numpy as np

def get_table_points(table_detections):
    left_table, right_table = table_detections

    left_table = np.array(left_table)[:, 0, :]
    right_table = np.array(right_table)[:, 0, :]

    left_center = left_table.mean(axis=0)
    right_center = right_table.mean(axis=0)

    middle_top_left_mask = (left_table[:, 0] > left_center[0]) & (left_table[:, 1] < left_center[1])
    middle_bottom_left_mask = (left_table[:, 0] > left_center[0]) & (left_table[:, 1] > left_center[1])
    middle_top_right_mask = (right_table[:, 0] < right_center[0]) & (right_table[:, 1] < right_center[1])
    middle_bottom_right_mask = (right_table[:, 0] < right_center[0]) & (right_table[:, 1] > right_center[1])

    # print(left_table.shape, right_table.shape, middle_top_left_mask, middle_top_right_mask, middle_bottom_left_mask, middle_bottom_right_mask)

    middle_top = (left_table[middle_top_left_mask] + right_table[middle_top_right_mask]) / 2
    middle_bottom = (left_table[middle_bottom_left_mask] + right_table[middle_bottom_right_mask]) / 2

    return middle_top[0], middle_bottom[0]

def get_net_offset(middle_top, middle_bottom, x, y):
    # middle_top[1] * s + middle_bottom[1] * (1 - s) = y
    # (middle_top[1] - middle_bottom[1]) * s + middle_bottom[1] = y
    s = (y - middle_bottom[1]) / (middle_top[1] - middle_bottom[1])

    cutoff_x = middle_bottom[0] * s + middle_top[0] * (1 - s)

    # 0 left, 1 right
    return x - cutoff_x

