import numpy as np
from Modules.TrackletManager import TrackletManager
from calcs import iou


class DataAssociationModule:
    TM: TrackletManager

    def __init__(self, t_max, lambda_iou=0.001, tracklet_manager=None):
        """
        Data Association Module
        :param t_max: max number of frames to keep a tracklet without detections alive
        :param lambda_iou: IOU threshold to connect a detection to a previous bbox
        """
        self.maxDisappeared = t_max
        self.lambda_iou = lambda_iou
        self.TM = tracklet_manager

    def update(self, input_bboxes, prev_frame=None, frame=None):
        # Check to see if the list of input bounding box rectangles is empty
        if len(input_bboxes) == 0:
            self.no_input(frame, prev_frame)
            # Return early as there are no positions to update
            return  # self.active_tracklets
        # If we are currently not tracking any objects take the input
        # boxes and register each one of them
        if len(self.TM.active_tracklets) == 0:
            for input_box in input_bboxes:
                self.TM.register(input_box)
        # Otherwise, we are currently tracking objects, so we need to
        # try to match the input bboxes to the existing object bboxes
        else:
            # Run IOU to match input boxes to existing objects' positions
            objectIDs, active_positions_array, input_positions_array, D, rows, cols = self.calc_iou_matches(input_bboxes)
            # Check if any active object won't have a match
            unmatched_rows, _ = self.check_possible_matches(rows, cols, D)
            # If there were unused rows, and these objects aren't already inactive
            # then we need to interpolate their lost position using a tracker, and also instantiate a tracker
            not_enough_matches = not(len(unmatched_rows) == 0)
            if not_enough_matches:
                # If there were tracklets which have not been matched, check and start their trackers
                newly_lost = self.start_tracking_lost_objects(objectIDs, unmatched_rows, frame, prev_frame)
                # If there has been a newly lost tracklet, we have to recalculate iou and try to re-match everything.
                # This is very expensive and the original paper does not explicitly say if it tries to re-match a lost
                # tracklet the same frame it was lost using the first tracker's prediction. All it says is: "In case the
                # detector loses a face in the following frame, it keeps predicting its position...".
                if newly_lost:
                    objectIDs, active_positions_array, input_positions_array, D, rows, cols = self.calc_iou_matches(
                        input_bboxes)
                    unused_rows, unused_cols = self.match_and_update(rows, cols, D, objectIDs, input_positions_array)
                # If there were no newly_lost tracklets then there's no need to recalculate iou, just match them
                else:
                    unused_rows, unused_cols = self.match_and_update(rows, cols, D, objectIDs, input_positions_array)
            # If every tracklet has been matched to an input box, just match and update them
            else:
                unused_rows, unused_cols = self.match_and_update(rows, cols, D, objectIDs, input_positions_array)
            # in the event that the number of object bboxes is
            # equal or greater than the number of input bboxes
            # we need to check and see if some of these objects have
            # potentially disappeared
            # loop over the unused row indexes
            for row in unused_rows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                tracklet_id = objectIDs[row]
                self.TM.active_tracklets[tracklet_id].disappeared_frames += 1
                self.TM.active_tracklets[tracklet_id].active = False
                # check to see if the number of consecutive
                # frames the tracklet has been marked "disappeared"
                # for warrants deregistering the object
                if self.TM.active_tracklets[tracklet_id].disappeared_frames > self.maxDisappeared:
                    self.TM.deregister(tracklet_id)
            # otherwise, if the number of input bboxes is greater
            # than the number of existing objects we need to
            # register each new input detection as a new tracklet
            for col in unused_cols:
                self.TM.register(input_positions_array[col])
        # return the set of trackable objects
        return self.TM.active_tracklets

    def no_input(self, curr_frame, prev_frame):
        # loop over any existing tracked objects and mark them
        # as disappeared
        for t_id in list(self.TM.active_tracklets.keys()):
            # If this tracklet has just been lost then start a tracker for it
            if self.TM.active_tracklets[t_id].active:
                # We need to start the tracker on its last known location, aka: the previous frame
                self.TM.active_tracklets[t_id].start_tracker(prev_frame)
                # Then we can update the tracker for the current frame
                self.TM.active_tracklets[t_id].update_tracker(curr_frame)
            # Mark it as inactive
            self.TM.active_tracklets[t_id].active = False
            # Add to the disappeared counter
            self.TM.active_tracklets[t_id].disappeared_frames += 1
            # if we have reached a maximum number of consecutive
            # frames where a given object has been marked as
            # missing, deregister it
            if self.TM.active_tracklets[t_id].disappeared_frames > self.maxDisappeared:
                self.TM.deregister(t_id)

    def calc_iou_matches(self, input_bboxes):
        # initialize an array of input bboxes for the current frame
        input_positions_array = np.array(input_bboxes)
        # grab the set of object IDs and corresponding centroids
        objectIDs = list(self.TM.active_tracklets.keys())
        active_positions_as_list = []
        for _, tcklet in self.TM.active_tracklets.items():
            active_positions_as_list.append(tcklet.position)
        active_positions_array = np.array(active_positions_as_list, dtype=np.int)

        # compute the distance between each pair of object
        # bounding boxes and input bounding boxes, respectively
        # -- our goal will be to match an input bbox to an
        D = iou(active_positions_array[:, None], input_positions_array[None])
        # in order to perform this matching we must (1) find the
        # smallest value in each row and then (2) sort the row
        # indexes based on their minimum values so that the row
        # with the smallest value is at the *front* of the index
        # list
        rows = D.max(axis=1).argsort()[::-1]
        # next, we perform a similar process on the columns by
        # finding the smallest value in each column and then
        # sorting using the previously computed row index list
        cols = D.argmax(axis=1)[rows]
        return objectIDs, active_positions_array, input_positions_array, D, rows, cols

    def check_possible_matches(self, rows, cols, iou_matrix):
        used_rows = set()
        used_cols = set()
        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
            # val
            if row in used_rows or col in used_cols or iou_matrix[row][col] <= self.lambda_iou:
                continue
            # indicate that we have examined each of the row and
            # column indexes, respectively
            used_rows.add(row)
            used_cols.add(col)
        # compute both the row and column index we have NOT yet
        # examined
        unused_rows = set(range(0, iou_matrix.shape[0])).difference(used_rows)
        unused_cols = set(range(0, iou_matrix.shape[1])).difference(used_cols)
        return unused_rows, unused_cols

    def start_tracking_lost_objects(self, obj_ids, unused_rows, curr_frame, prev_frame):
        # Each row index in unused_rows list is an object that hasn't been matched this frame
        # therefore we need to pass by each one and check if it was previously active.
        newly_lost = False
        for row in unused_rows:
            # grab the object ID for the corresponding row index
            tracklet_id = obj_ids[row]
            # If this is the first frame where the tracklet is lost then start a tracker
            if self.TM.active_tracklets[tracklet_id].active:
                # We need to start the tracker on it's last known location, aka: the previous frame
                self.TM.active_tracklets[tracklet_id].start_tracker(prev_frame)
                # Then we can update the tracker for the current frame
                self.TM.active_tracklets[tracklet_id].update_tracker(curr_frame)
                # Flag that there was a newly lost tracklet
                newly_lost = True
            # If this tracklet is already inactive, then its tracker was already updated for
            # the current frame.
            else:
                continue
        return newly_lost

    def match_and_update(self, rows, cols, D, obj_ids, input_positions_array):
        # in order to determine if we need to update, register,
        # or deregister an object we need to keep track of which
        # of the rows and column indexes we have already examined
        usedRows = set()
        usedCols = set()
        # loop over the combination of the (row, column) index
        # tuples
        for (row, col) in zip(rows, cols):
            # if we have already examined either the row or
            # column value before, ignore it
            # val
            if row in usedRows or col in usedCols or D[row][col] <= self.lambda_iou:
                continue
            # otherwise, grab the object ID for the current row,
            # set its new centroid, and reset the disappeared
            # counter
            tracklet_id = obj_ids[row]
            self.TM.active_tracklets[tracklet_id].set_position(input_positions_array[col])
            self.TM.active_tracklets[tracklet_id].disappeared_frames = 0
            self.TM.active_tracklets[tracklet_id].active = True
            # indicate that we have examined each of the row and
            # column indexes, respectively
            usedRows.add(row)
            usedCols.add(col)
        # compute both the row and column index we have NOT yet
        # examined
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)
        return unusedRows, unusedCols
