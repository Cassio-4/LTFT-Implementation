# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from torch.nn import DataParallel
from Arcface.models.resnet import resnet_face18
# Link to download pretrained model https://drive.google.com/file/d/1m7jSX3RLkhEWs_2Xkr4T76PTcqMHFLaO/view?usp=sharing


class Arcface:
    def __init__(self, arcface_dict, use_se=False):
        self.model = resnet_face18(use_se)
        self.model = DataParallel(self.model)
        self.mode = arcface_dict["mode"]
        if self.mode == "cpu":
            self.model.load_state_dict(torch.load(arcface_dict["model_path"], map_location="cpu"))
        else:
            self.model.load_state_dict(torch.load(arcface_dict["model_path"]))
        self.model.to(torch.device(self.mode))
        self.model.eval()

        #self.threshold = arcface_dict["threshold"]
        self.batch_size = arcface_dict["batch_size"]

    def get_single_feature(self, image):
        """
        Gets the feature vector for a single face image
        :param image: ndarray face image
        :return: feature vector
        """
        image = treat_image(image)
        if image is None:
            exit(1)

        data = torch.from_numpy(image)
        data = data.to(torch.device(self.mode))
        output = self.model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))

        return feature


def treat_image(image):
    """
    Gets an image ndarray and treats it as needed for the arcface face recognition model
    :param image: ndarray crop of face
    :return: treated image
    """
    # image = cv2.imread(img_path, 0)
    #   cv::IMREAD_GRAYSCALE = 0,
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    if image is None:
        # TODO Treat this error if it ever happens
        print("Image is none")
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)
        #print(images.shape)
        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cpu"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
