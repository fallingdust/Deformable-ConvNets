import math


def cal_rotated_bbox(x1, y1, x2, y2, rotated_degree, center_x, center_y):
    x3 = x1
    y3 = y2
    x4 = x2
    y4 = y1
    x1 -= center_x
    y1 -= center_y
    x2 -= center_x
    y2 -= center_y
    x3 -= center_x
    y3 -= center_y
    x4 -= center_x
    y4 -= center_y
    angle = rotated_degree * math.pi / 180
    a = math.cos(angle)
    b = math.sin(angle)
    p0_x = x1 * a + y1 * b + center_x
    p0_y = y1 * a - x1 * b + center_y
    p1_x = x2 * a + y2 * b + center_x
    p1_y = y2 * a - x2 * b + center_y
    p2_x = x3 * a + y3 * b + center_x
    p2_y = y3 * a - x3 * b + center_y
    p3_x = x4 * a + y4 * b + center_x
    p3_y = y4 * a - x4 * b + center_y
    x1_ = min(max(int(math.floor(min(p0_x, p1_x, p2_x, p3_x))), 0), 2 * center_x)
    y1_ = min(max(int(math.floor(min(p0_y, p1_y, p2_y, p3_y))), 0), 2 * center_y)
    x2_ = min(max(int(math.ceil(max(p0_x, p1_x, p2_x, p3_x))), 0), 2 * center_x)
    y2_ = min(max(int(math.ceil(max(p0_y, p1_y, p2_y, p3_y))), 0), 2 * center_y)
    return [x1_, y1_, x2_, y2_]