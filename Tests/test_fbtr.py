from modules.fbtr import FaceBasedTrackletReconnectionModule, get_blur_metric, get_7_landmarks
from DDFA.utils.inference import draw_landmarks
import cv2
from config import config_dict

pics_folder = "pics/"
test_dets_and_scores = [["Lenna_0.jpg", "Henry-Cavill_0.jpg", "Mot17-09_000123_0.jpg", "Mot17-09_000123_1.jpg",
                        "Mot17-09_000123_2.jpg", "Mot17-09_000123_3.jpg", "Mot17-09_000461_0.jpg",
                        "Mot17-09_000461_1.jpg", "bengal_228_0.jpg", "bengal_228_1.jpg", "bengal_228_2.jpg",
                        "bengal_228_3.jpg", "choke1_1374_0.jpg", "choke1_1374_1.jpg", "choke1_1374_2.jpg",
                        "choke1_1374_3.jpg", "choke1_1374_4.jpg", "choke1_1374_5.jpg", "choke1_1543_0.jpg",
                        "choke1_1543_1.jpg", "choke1_1543_2.jpg", "choke1_1543_3.jpg", "choke1_1543_4.jpg",
                        "choke1_1543_5.jpg", "choke1_696_0.jpg", "choke1_696_1.jpg", "choke1_696_2.jpg",
                        "choke1_696_3.jpg"],
                        [0.87, 0.9, 0.85, 0.85, 0.83, 0.71, 0.85, 0.72, 0.86, 0.86, 0.74, 0.72, 0.89, 0.86, 0.86, 0.86,
                         0.85, 0.85, 0.87, 0.86, 0.85, 0.84, 0.77, 0.73, 0.88, 0.87, 0.86, 0.84]]

fbtr = FaceBasedTrackletReconnectionModule(config_dict["3ddfa_config"], config_dict["arcface_config"])
pi = 3.141592653589793
for pic_name, score in zip(test_dets_and_scores[0], test_dets_and_scores[1]):
    img = cv2.imread(pics_folder + pic_name)
    face_quality_group = fbtr.get_quality_indicator(img, score)
    print("{}'s quality is -> {}".format(pic_name, face_quality_group))
    # This only draws the landmarks on the picture and writes it to disk for verification, since the Sharpness Score
    # is yet to be implemented, we won't be using the landmarks for now.
    """
    img_points = img.copy()
    for i in range(0, len(pts68[0])):
        point = (int(pts68[0][i]), int(pts68[1][i]))
        img_points = cv2.circle(img_points, point, 1, (0, 255, 0))
    cv2.imwrite("lenna_my_points.jpg", img_points)
    pts = get_7_landmarks(pts68)
    blur = get_blur_metric(img, pts)
    pass
    """
#TODO test other pics
pass

