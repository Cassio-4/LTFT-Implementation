import numpy as np

low = np.s_[..., :2]
high = np.s_[..., 2:]


def iou(A, B):
    A, B = A.copy(), B.copy()
    A[high] += 1
    B[high] += 1
    intrs = (np.maximum(0, np.minimum(A[high], B[high])
                        - np.maximum(A[low], B[low]))).prod(-1)
    return intrs / ((A[high] - A[low]).prod(-1) + (B[high] - B[low]).prod(-1) - intrs)


def centroid_from_bbox(rect):
    """
    calculate centroid of a bounding box
    :param rect: 4 numbers corresponding to a box coordinate [xmin, ymin, xmax, ymax]
    :return: a coordinate tuple (x, y)
    """
    # use the bounding box coordinates to derive the centroid
    cx = int((rect[0] + rect[2]) / 2.0)
    cy = int((rect[1] + rect[3]) / 2.0)
    return (cx, cy)
