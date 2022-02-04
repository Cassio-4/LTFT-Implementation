from typing import Dict
import numpy as np
import cv2


class Tracklet:
    tracker: cv2.TrackerKCF

    def __init__(self, id, bbox, det_score=0.0):
        """
        Tracklet class
        :param id: object id
        :param bbox: four ints in format (xmin, ymin, xmax, ymax)
        """
        # store this tracklet's ID
        self.id = id
        # Store its position when created
        self.position = np.array(bbox, dtype=np.uint)
        # Number of frames in which the object has disappeared and is being tracked by
        # an object tracker
        self.disappeared_frames = 0
        # Boolean flag for if the tracklet is active or not
        self.active = True
        # Tracker to interpolate position when detector fails
        self.tracker = None
        # Mean template of features obtained from ArcFace's model, using faces with verifiable quality
        self.verifiable_features_running_mean = None
        self.verifiable_count = 0
        # Mean template of features obtained from ArcFace's model, using faces with enrollable quality
        self.enrollable_features_running_mean = None
        self.enrollable_count = 0
        # Last detection's score
        self.latest_score = det_score
        # Ordered dictionary of [int, array] where key is frame number and array is the tracklet position
        # [xmin, ymin, xmax, ymax]
        self.position_history = {}

    def set_position(self, new_box):
        """
        Sets the new position for this tracklet
        :param new_box: any list/array/tuple of 4 uint values [xmin, ymin, xmax, ymax]
        :return:
        """
        self.position[0] = new_box[0]
        self.position[1] = new_box[1]
        self.position[2] = new_box[2]
        self.position[3] = new_box[3]

    def start_tracker(self, frame):
        p = cv2.TrackerKCF_Params()
        p.detect_thresh = 0.01
        self.tracker = cv2.TrackerKCF_create(p)
        bbox = (self.position[0], self.position[1], self.position[2]-self.position[0], self.position[3]-self.position[1])
        self.tracker.init(frame, bbox)

    def update_tracker(self, frame):
        (success, box) = self.tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.set_position(np.array((x, y, x+w, y+h), dtype=np.uint))
        else:
            # The paper does not say what to do when kcf fails
            # do nothing
            pass

    def update_mean_verifiable(self, new_features):
        if self.verifiable_features_running_mean is None:
            self.verifiable_features_running_mean = np.zeros(1024, dtype=np.float64)
        self.verifiable_count += 1
        n = float(self.verifiable_count)

        # Calculate running mean
        #for x_i in x:
        # formula: m = m + (x_i - m) / n
        new_features = np.subtract(new_features, self.verifiable_features_running_mean)
        np.true_divide(new_features, n, out=new_features)
        np.add(self.verifiable_features_running_mean, new_features, out=self.verifiable_features_running_mean,
               dtype=np.float64)

    def update_mean_enrollable(self, new_features):
        if self.enrollable_features_running_mean is None:
            self.enrollable_features_running_mean = np.zeros(1024, dtype=np.float64)
        self.enrollable_count += 1
        n = float(self.enrollable_count)

        # Calculate running mean
        #for x_i in x:
        # formula: m = m + (x_i - m) / n
        new_features = np.subtract(new_features, self.enrollable_features_running_mean)
        np.true_divide(new_features, n, out=new_features)
        np.add(self.enrollable_features_running_mean, new_features, out=self.enrollable_features_running_mean,
               dtype=np.float64)

    def add_to_position_history(self, frame_num):
        self.position_history[frame_num] = self.position.copy()



class TrackletManager:
    active_tracklets: Dict[int, Tracklet]

    def __init__(self):
        self.active_tracklets = {}
        self.next_id = 0
        self.inactive_tracklets = {}

    def register(self, box, score):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.active_tracklets[self.next_id] = Tracklet(self.next_id, box, score)
        self.next_id += 1

    def deregister(self, id):
        # To deregister a tracklet we pop it from active_tracklets dict
        # and move it to inactive tracklets dict
        popped_tracklet = self.active_tracklets.pop(id)
        self.inactive_tracklets[id] = popped_tracklet

    def add_to_tracklets_histories(self, frame_num):
        for _, tracklet in self.active_tracklets.items():
            tracklet.add_to_position_history(frame_num)
