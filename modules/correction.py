from typing import Tuple
from .TrackletManager import TrackletManager


class CorrectionModule:
    tm: TrackletManager

    def __init__(self, tracklet_manager):
        self.tm = tracklet_manager

    def correct_pairs(self, list_of_pairs):
        """
        Fuses the pairs of tracklets (Tk, Ti) into one tracklet
        :type list_of_pairs: Tuple[int, int]
        """
        if list_of_pairs is None:
            # No pairs to connect
            return
        for pair in list_of_pairs:
            tk_id, ti_id = pair
            try:
                tk = self.tm.active_tracklets.pop(tk_id)
            except KeyError:
                print("KeyError obtaining tk from active_tracklets")
                exit(1)
            try:
                ti = self.tm.active_tracklets.pop(ti_id)
            except KeyError:
                ti = self.tm.inactive_tracklets.pop(ti_id)
            # For each pair (Tk, Ti), it retrieves all the detections assigned in the past to Tk and
            # switches their track ID to Ti.
            ti.position_history.update(tk.position_history)
            tk.position_history = ti.position_history
            tk.id = ti_id
            # The paper doesn't say what to do with the mean of face templates, so, I am keeping the one with
            # most samples
            if tk.verifiable_count < ti.verifiable_count:
                tk.verifiable_features_running_mean = ti.verifiable_features_running_mean
                tk.verifiable_count = ti.verifiable_count
            if tk.enrollable_count < ti.enrollable_count:
                tk.enrollable_features_running_mean = ti.enrollable_features_running_mean
                tk.enrollable_count = ti.enrollable_count

            # Finally, reinsert Tracklet into active tracklets dict
            if tk.id in self.tm.active_tracklets.keys():
                print("Tracklet with id {} already exists, should've been popped".format(tk.id))
            self.tm.active_tracklets[tk.id] = tk
