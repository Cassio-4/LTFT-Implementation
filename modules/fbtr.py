import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import cv2
# This will not work unless you have created a symlink "DDFA" between the LTFT project root and 3DDFA submodule
from DDFA.utils.inference import parse_roi_box_from_landmark, parse_roi_box_from_bbox, predict_68pts, crop_img
from DDFA.utils.estimate_pose import parse_pose
from DDFA.utils.ddfa import NormalizeGjz, ToTensorGjz
from DDFA import mobilenet_v1
from .arcface import Arcface
from scipy.spatial.distance import cdist


class FaceBasedTrackletReconnectionModule:
    def __init__(self, fbtr_dict, arcface_dict):
        print("Loading 3DDFA pose estimation model")
        self._transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        self.mode = fbtr_dict["3ddfa_mode"]
        self.STD_SIZE = 120
        self.bbox_init = fbtr_dict["3ddfa_bbox_init"]
        self.model_3ddfa = self.load_3ddfa_model(fbtr_dict["3ddfa_model_path"])
        print("Done loading 3DDFA model.")
        print("Loading ArcFace model")
        self._arcface = Arcface(arcface_dict)
        print("Done loading Arcface model.")
        self.lambda_recognition = fbtr_dict["recognition_threshold"]
        self.blur_higher_thresh, self.blur_lower_thresh = fbtr_dict["blur_thresholds"]
        self._e = fbtr_dict["e_margin"]
        self._C = fbtr_dict["C"]

    def forward_3ddfa(self, face_images):
        """
        Pass image through the 3DDFA model to get pose and landmark points
        :param face_images: A cropped face detection
        """

        for face_det in face_images:
            # This pipeline doesn't need to load image from file, we're already providing face detections
            ind = 0

            # Since we are using already cropped images, our roi box is the dimensions of the picture
            row_num, col_num, layer_num = face_det.shape
            bbox = [0.0, 0.0, col_num - 1, row_num - 1]  # [rect.left(), rect.top(), rect.right(), rect.bottom()]
            roi_box = parse_roi_box_from_bbox(bbox)
            # This would be the crop of the face, however our face images should already arrive cropped
            img = crop_img(face_det, roi_box)
            # forward: one step
            img = cv2.resize(img, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = self._transform(img).unsqueeze(0)
            with torch.no_grad():
                if self.mode == 'gpu':
                    input = input.cuda()
                param = self.model_3ddfa(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if self.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(face_det, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = self._transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if self.mode == 'gpu':
                        input = input.cuda()
                    param = self.model_3ddfa(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            # pts_res.append(pts68)
            P, pose = parse_pose(param)

            return pose, pts68

    def calculate_quality_all_dets(self, tracklet_manager, frame):
        """
        Iterates through all active tracklets and qualifies its latest detection in one of three categories,
        Enrollable, Verifiable or Discarded, then updates the feature mean template
        :return: Nothing
        """
        for _, tcklet in tracklet_manager.active_tracklets.items():
            if tcklet.active:
                xmin, ymin, xmax, ymax = tcklet.position[0:4]
                face_detection_crop = frame[ymin:ymax, xmin:xmax]
                # Get quality indicator
                quality_group = self.get_quality_indicator(face_detection_crop, tcklet.latest_score)
                # If discarded
                if quality_group == 2:
                    continue
                # If previous if didn't skip, then quality is verifiable or enrollable, calc features and save
                feat = np.squeeze(self._arcface.get_single_feature(face_detection_crop))
                # If verifiable
                if quality_group == 1:
                    tcklet.update_mean_verifiable(feat)
                # Else, it is Enrollable
                else:
                    tcklet.update_mean_enrollable(feat)

    def get_quality_indicator(self, face_det_image, score):
        """
        Gets all quality metrics and returns which category the face detection belongs to.
        :param face_det_image: face image array in BGR format (opencv format)
        :param score: the confidence score provided by the detector
        :return: 0 if enrollable, 1 if verifiable or 2 if discarded
        """
        # Mr. Barquero said all face images were resized to same size before sharpness measure and quality assessment
        resized = cv2.resize(face_det_image, (120, 120), cv2.INTER_LINEAR)
        pose, pts = self.forward_3ddfa([resized])
        # Convert pose from radians to degrees
        degrees = [round(((rad * 180) / 3.141592653589793), 2) for rad in pose]
        # Get range of pose
        in_60_range = [True if (-60.0 <= x <= 60.0) else False for x in degrees]
        # If detection score > 0.8 and pose range between +-60 degrees it could be enrollable or verifiable
        if score > 0.8 and not (False in in_60_range):
            # Get blur score, we only need to calculate it if there's a chance for the face to be either enrollable
            # or verifiable
            blur_score = get_blur_metric(face_det_image, pts)
            #print("blur_score = {}".format(blur_score))
            # Check if pose range between +- 25 degrees
            in_25_range = [True if (-25.0 <= x <= 25.0) else False for x in degrees]
            # Knowing the range of pose and blur measure, if enrollable
            if score >= 0.95 and not (False in in_25_range) and blur_score >= self.blur_higher_thresh:
                return 0
            # If not enrollable, is it verifiable?
            elif blur_score >= self.blur_lower_thresh:
                return 1
        # If it didn't return previously then definitely is a discarded face
        return 2

    def compute_face_similarities(self, tracklet_manager):
        # {Tk} set
        active_tracklet_templates = []
        active_tracklet_ids = []
        # {Ti} belonging to T \ {Tk}
        inactive_tracklet_templates = []
        inactive_tracklet_ids = []
        for id, tracklet in tracklet_manager.active_tracklets.items():
            # Then, for each tracklet Tk with an assigned detection Dk in the current frame
            if tracklet.active:
                if tracklet.verifiable_features_running_mean is None:
                    continue
                active_tracklet_ids.append(id)
                # The mean of the verifiable face templates of Tk, VTk, is also computed.
                active_tracklet_templates.append(tracklet.verifiable_features_running_mean)
            else:
                if tracklet.enrollable_features_running_mean is None:
                    continue
                inactive_tracklet_ids.append(id)
                # ...we retrieve tracklets Ti with i != k. For each tracklet Ti, the mean of its enrollable
                # face templates, ETi, is computed and taken as the tracklet reference template.
                inactive_tracklet_templates.append(tracklet.enrollable_features_running_mean)
        # Get all other Ti tracklets
        for id, tracklet in tracklet_manager.inactive_tracklets.items():
            if tracklet.enrollable_features_running_mean is None:
                continue
            inactive_tracklet_ids.append(id)
            # ...we retrieve tracklets Ti with i != k. For each tracklet Ti, the mean of its enrollable
            # face templates, ETi, is computed and taken as the tracklet reference template.
            inactive_tracklet_templates.append(tracklet.enrollable_features_running_mean)

        if (not active_tracklet_templates) or (not inactive_tracklet_templates):
            # One of the lists is empty, there's nothing to calculate, quit method
            return

        x_k = np.array(active_tracklet_templates)
        x_i = np.array(inactive_tracklet_templates)
        distances = cdist(x_k, x_i, 'cosine')
        # Turn distance into similarity
        distances = 1 - distances
        # Check the distances and see if any beat the threshold of recognition
        rows = distances.max(axis=1).argsort()[::-1]
        cols = distances.argmax(axis=1)[rows]
        used_rows = set()
        used_cols = set()
        # loop over the combination of the (row, column) index
        # tuples
        list_of_pairs = []
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or column value;
            # or if the similarity measure is lower than the threshold lambdaFBTR;
            # or if there are less than C other values to get average, we skip
            if row in used_rows or col in used_cols or distances[row][col] <= self.lambda_recognition \
                    or len(distances[row]) < self._C+1:
                continue
            # That highest similarity is above the average of the next C-highest ones, by a margin 1/e:
            # Get C next highest ones
            C_highest_average = 0.0
            sorted_indexes = distances[row].argsort()[::-1][:self._C + 1]
            for idx in sorted_indexes[1:]:
                C_highest_average += distances[row][idx]
            # Get average of C highest values
            C_highest_average = C_highest_average / self._C
            # multiply it by 1/e
            C_highest_average = C_highest_average * (1 / self._e)
            # Finally if the highest similarity is above the average of the C_highest_average then we connect them
            if distances[row][col] >= C_highest_average:
                list_of_pairs.append((active_tracklet_ids[row], inactive_tracklet_ids[col]))
                # indicate that we have examined each of the row and
                # column indexes, respectively
                used_rows.add(row)
                used_cols.add(col)
            # Else, just continue the loop
        # Return list of connected pairs with the highest similarity, above the C-highest average
        return list_of_pairs

    def load_3ddfa_model(self, path):
        checkpoint_fp = path
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        if self.mode == "gpu":
            cudnn.benchmark = True
            model = model.cuda()
        model.eval()
        return model


# Implementation of section 2.2 of https://gc2011.graphicon.ru/html/2014/papers/111-114.pdf
def get_blur_metric(face_image, landmarks):
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
    # --- end of laplace ---

    # Build a mask to remove background information
    landmarks = get_7_landmarks(landmarks)
    mask = np.zeros(lap.shape[:2], np.uint8)
    mask = cv2.fillConvexPoly(mask, landmarks, 1, cv2.LINE_8)

    # The paper says: "The image sharpness score is calculated as the averaged Laplace
    # operator response by masked image".
    values = lap[np.where(mask != 0)]
    if values.size != 0:
        blur_score = float(np.mean(values))
        return round(blur_score, 4)
    else:
        return 0.0


def get_7_landmarks(pts68):
    pts = np.array([[pts68[0][36], pts68[1][36]], [pts68[0][19], pts68[1][19]], [pts68[0][24], pts68[1][24]],
                    [pts68[0][45], pts68[1][45]], [pts68[0][54], pts68[1][54]], [pts68[0][57], pts68[1][57]],
                    [pts68[0][48], pts68[1][48]]], dtype=np.uint)
    return pts
