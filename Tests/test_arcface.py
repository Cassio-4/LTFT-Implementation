from modules.arcface import Arcface, cosin_metric
from config import arcface_config
import cv2
import numpy as np
from scipy.spatial.distance import cdist
arc = Arcface(arcface_config)
features_list = []
pics = [cv2.imread("./pics/Lenna_0.jpg"), cv2.imread("./pics/choke1_1543_0.jpg"), cv2.imread("./pics/choke1_696_1.jpg"),
        cv2.imread("./pics/choke1_696_0.jpg"), cv2.imread("./pics/choke_test0.jpg"), cv2.imread("./pics/choke1_1543_1.jpg")]
for img in pics:
    features_list.append(arc.get_single_feature(img))
# ---
for feature_index in range(len(features_list)):
    features_list[feature_index] = np.reshape(np.squeeze(features_list[feature_index]), 1024)
# ---
for i in range(len(features_list)):
    for j in range(len(features_list)):
        cm = cosin_metric(features_list[i], features_list[j])
        print("{} -> {} == {}".format(i, j, cm))
# ---
x_k = np.array(features_list)
distances = cdist(x_k, x_k, 'cosine')
distances = 1 - distances
pass