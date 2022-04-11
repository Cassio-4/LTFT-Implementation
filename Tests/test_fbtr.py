from modules.fbtr import FaceBasedTrackletReconnectionModule, get_7_landmarks
import cv2
from config import config_dict
import numpy as np


def lowpass_blur(face_image, img_name):
    lowpass = cv2.blur(face_image, (3, 3))
    absolute = np.abs(face_image - lowpass)
    avg = np.average(absolute)
    print("{}'s score is -> {}".format(img_name, avg))


def get_blur_metric(face_image, landmarks, img_name):
    # Convert image to grayscale single channel
    image_ = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Convert image to double and in range [0.0, 1.0], Matlab just divides an uint8 image by 255
    image_ = np.divide(image_, 255.0, dtype=np.single)
    # --- Operate Laplace on whole image ---
    # Use Laplace filter to obtain contours and edges, the more sharp an image is the greater the response
    # from the laplace filter
    filter1 = np.array([[1, -2, 1]])
    filter2 = np.array([[1], [-2], [1]])
    pass1 = cv2.filter2D(image_, cv2.CV_32F, filter1, borderType=cv2.BORDER_CONSTANT)
    pass2 = cv2.filter2D(image_, cv2.CV_32F, filter2, borderType=cv2.BORDER_CONSTANT)
    lap = cv2.add(np.abs(pass1), np.abs(pass2), dtype=cv2.CV_32F)
    #print("{} sum -> {}".format(img_name, round(float(np.sum(lap)), 4)))
    # --- end of laplace ---

    # --- get mask to remove background ---
    landmarks = get_7_landmarks(landmarks)
    mask = np.zeros(lap.shape[:2], np.uint8)
    mask = cv2.fillConvexPoly(mask, landmarks, 1, cv2.LINE_8)

    # The paper says: "The image sharpness score is calculated as the averaged Laplace
    # operator response by masked image".
    values = lap[np.where(mask != 0)]
    blur_score = float(np.mean(values))
    return round(blur_score, 4)


pics_folder = "pics/"
test_dets_and_scores = [["choke_test0.jpg", "Lenna_0.jpg", "Henry-Cavill_0.jpg", "Mot17-09_000123_0.jpg", "Mot17-09_000123_1.jpg",
                        "Mot17-09_000123_2.jpg", "Mot17-09_000123_3.jpg", "Mot17-09_000461_0.jpg",
                        "bengal_228_0.jpg", "bengal_228_1.jpg", "choke1_1374_0.jpg", "choke1_1374_1.jpg",
                        "choke1_1374_2.jpg",
                        "choke1_1374_3.jpg", "choke1_1374_4.jpg", "choke1_1374_5.jpg", "choke1_1543_0.jpg",
                        "choke1_1543_1.jpg", "choke1_1543_2.jpg", "choke1_1543_3.jpg",
                        "choke1_696_0.jpg", "choke1_696_1.jpg", "choke1_696_2.jpg",
                        "choke1_696_3.jpg", "choke2_688.jpg"],
                        [0.9, 0.87, 0.9, 0.85, 0.85, 0.83, 0.71, 0.85, 0.86, 0.86, 0.89, 0.86, 0.86, 0.86,
                         0.85, 0.85, 0.87, 0.86, 0.85, 0.84, 0.88, 0.87, 0.86, 0.84, 0.95]]

fbtr = FaceBasedTrackletReconnectionModule(config_dict["fbtr_config"], config_dict["arcface_config"])
"""
img = cv2.imread(pics_folder + "choke_test0.jpg")
pose, pts = fbtr.forward_3ddfa([img])
blur_score = get_blur_metric(img, pts, "choke_test0")
print("choke_test0", blur_score)

img = cv2.imread(pics_folder + "bengal_228_3.jpg")
img = cv2.resize(img, (120, 120), cv2.INTER_LINEAR)
pose, pts = fbtr.forward_3ddfa([img])
blur_score = get_blur_metric(img, pts, "bengal_228_3")
print("bengal_228_3 ", blur_score)
"""

for pic_name, score in zip(test_dets_and_scores[0], test_dets_and_scores[1]):
    img = cv2.imread(pics_folder + pic_name)
    #face_quality_group = fbtr.get_quality_indicator(img, score)
    #print("{}'s quality is -> {}".format(pic_name, face_quality_group))
    img = cv2.resize(img, (120, 120), cv2.INTER_LINEAR)
    pose, pts = fbtr.forward_3ddfa([img])
    degrees = [round(((rad * 180) / 3.141592653589793), 2) for rad in pose]
    blur_score = get_blur_metric(img, pts, pic_name)
    #print("img: {} has degrees = {}".format(pic_name, degrees))
    print("img: {} has blur score = {}".format(pic_name, blur_score))
    lowpass_blur(img, pic_name)
pass
#lowpass_blur(cv2.resize(cv2.imread(pics_folder+"bengal_228_0_yolo.jpg"), (120, 120)), "bengal_yolo")

