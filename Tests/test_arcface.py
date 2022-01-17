from Modules.arcface import Arcface, cosin_metric
import cv2
import numpy as np

arc = Arcface("../resnet18_110.pth", "cpu")

pic = cv2.imread("./pics/Lenna_0.jpg")
pic1 = cv2.imread("./pics/choke1_1543_0.jpg")
pic2 = cv2.imread("./pics/choke1_696_1.jpg")
feat = arc.get_single_feature(pic)
feat1 = arc.get_single_feature(pic1)
feat2 = arc.get_single_feature(pic2)
feat1 = np.squeeze(feat1)
feat2 = np.squeeze(feat2)
feat = np.squeeze(feat)
print(cosin_metric(feat1, feat2))
print(cosin_metric(feat1, feat))
print(cosin_metric(feat2, feat1))
pass