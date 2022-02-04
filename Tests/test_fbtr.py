from modules.fbtr import FaceBasedTrackletReconnectionModule, get_7_landmarks
import cv2
from config import config_dict
import numpy as np


def get_blur_metric(face_image, landmarks, img_name):
    # Changing the order of masked image
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # --- Operate Laplace on whole image ---
    # Use Laplace filter to obtain contours and edges, the more sharp an image is the greater the response
    # from the laplace filter
    kernel1 = np.array([[1, -2, 1]])
    kernel2 = np.array([[1],
                        [-2],
                        [1]])
    pass1 = cv2.filter2D(face_image, cv2.CV_32F, kernel1)
    pass2 = cv2.filter2D(face_image, cv2.CV_32F, kernel2)
    pass1 = np.abs(pass1)
    pass2 = np.abs(pass2)
    lap = cv2.add(pass1, pass2, dtype=cv2.CV_8U)
    if "choke2_688" in img_name:
        cv2.imwrite("laplace{}.jpg".format(img_name), lap)
    # --- end of laplace ---

    # --- get crop ---
    landmarks = get_7_landmarks(landmarks)
    # Get an even smaller bbox from the landmark points
    rect = cv2.boundingRect(landmarks)
    x, y, w, h = rect
    # Crop the face from the image using the landmark points bbox
    cropped = lap[y:y + h, x:x + w].copy()
    # Build a mask to remove background information
    pts = landmarks - landmarks.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # Use mask to crop face using bitwise and operation
    final = cv2.bitwise_and(cropped, cropped, mask=mask)
    if "choke2_688" in img_name:
        cv2.imwrite("final{}".format(img_name), final)


    # Equalize histogram
    #lap = cv2.cvtColor(lap, cv2.COLOR_BGR2GRAY)
    #lap = np.power(lap, 3)
    #final = cv2.equalizeHist(final)

    #cv2.imwrite("power_{}".format(img_name), final)

    # The paper where this idea originated from says: "The image
    # sharpness score is calculated as the averaged Laplace
    # operator response by masked image".
    values = final[np.where(mask != 0)]
    pixels_sum = np.sum(values)

    blur_custom = pixels_sum / values.size
    slope = (1.0 - 0.0) / (255 - 0)
    output = 0.0 + slope * (blur_custom - 0)
    blur_custom = round(output, 2)
    return blur_custom


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

fbtr = FaceBasedTrackletReconnectionModule(config_dict["3ddfa_config"], config_dict["arcface_config"])
"""
img = cv2.imread(pics_folder + "Henry-Cavill_0.jpg")
img = cv2.resize(img, (120, 120), cv2.INTER_LINEAR)
pose, pts = fbtr.forward_3ddfa([img])
blur_score = get_blur_metric(img, pts, "henry_cavill")
print("henry ", blur_score)

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
    #img = cv2.resize(img, (120, 120), cv2.INTER_LINEAR)
    pose, pts = fbtr.forward_3ddfa([img])
    blur_score = get_blur_metric(img, pts, pic_name)
    print("img: {} has blur score = {}".format(pic_name, blur_score))
pass
