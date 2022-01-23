from modules.TrackletManager import TrackletManager


# The tracking module is responsible for running the tracker
# when the detector fails. In other words, when the data association
# module fails to match detections to existing objects, the
# tracking module is responsible for updating the missing object's
# positions through a specific tracker (KCF originally)
class TrackingModule:
    tracklet_mngr: TrackletManager

    def __init__(self, tracklet_manager):
        self.tracklet_mngr = tracklet_manager

    # If there are objects already missing, then we need to update their bounding
    # boxes position for the current frame before matching
    def update_missing_objects_tracker(self, frame):
        for _, tcklet in self.tracklet_mngr.active_tracklets.items():
            if not tcklet.active:
                tcklet.update_tracker(frame)

    # If, at the end of matching the input bboxes to the already existing objects
    def track_missing_objects(self, prev_frame, frame, unused_row, unused_col, d):
        pass