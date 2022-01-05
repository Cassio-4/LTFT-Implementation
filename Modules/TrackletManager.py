from typing import Dict
import numpy as np
import cv2


class Tracklet:
    tracker: cv2.TrackerKCF

    def __init__(self, id, bbox):
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
        # Faces detected that belong to this tracklet
        self.face_detections = []
        # Tracker to interpolate position when detector fails
        self.tracker = None

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
        self.tracker = cv2.TrackerKCF_create()
        bbox = (self.position[0], self.position[1], self.position[2]-self.position[0], self.position[3]-self.position[1])
        self.tracker.init(frame, bbox)

    def update_tracker(self, frame):
        # TODO tracklet 11 has none as tracker, when calling update, find frame find why just run debug and you will remember
        (success, box) = self.tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            self.set_position(np.array((x, y, x+w, y+h), dtype=np.uint))
        else:
            # TODO treat this exception if it ever occurs
            # The paper does not say what to do when kcf fails
            # do nothing
            pass


class TrackletManager:
    active_tracklets: Dict[int, Tracklet]

    def __init__(self):
        self.active_tracklets = {}
        self.next_id = 0
        self.inactive_tracklets = {}

    def register(self, box):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.active_tracklets[self.next_id] = Tracklet(self.next_id, box)
        self.next_id += 1

    def deregister(self, id):
        # To deregister a tracklet we pop it from active_tracklets dict
        # and move it to inactive tracklets dict
        popped_tracklet = self.active_tracklets.pop(id)
        self.inactive_tracklets[id] = popped_tracklet