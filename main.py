from utils import ready_frames_and_detections, image_resize, draw_boxes_and_ids, write_results_mot_format
from modules.fbtr import FaceBasedTrackletReconnectionModule
from modules.data_association import DataAssociationModule
from modules.TrackletManager import TrackletManager
from modules.tracking import TrackingModule
from modules.correction import CorrectionModule
from config import config_dict
import cv2


if __name__ == '__main__':
    # Ready everything needed
    all_frames_and_dets = ready_frames_and_detections(frames_folder="/home/cassio/dataset/Images/MOT17-09",
                                                      dets_folder="/home/cassio/CrowdedDataset/Yolov5s-nothresh-clean"
                                                                  "-scoxyXY")
    # Initialize Tracklet Manager
    manager = TrackletManager()
    # Initialize every module
    tm = TrackingModule(tracklet_manager=manager)
    da = DataAssociationModule(config_dict["data_association_config"], tracklet_manager=manager)
    fbtr = FaceBasedTrackletReconnectionModule(config_dict["3ddfa_config"], config_dict["arcface_config"])
    cm = CorrectionModule(manager)
    # main loop
    count = 1
    prev_frame = None
    for img_path, detections in all_frames_and_dets:
        # Read frame
        frame = cv2.imread(img_path)
        # Ready all detections on current frame
        rects = []
        scores = []
        for det in detections:
            rects.append(det[1:])
            scores.append(det[0])
        # ========== Tracking Pipeline ==========
        # First update the positions for the objects that haven't been previously matched
        # thus are being tracked via specific tracking algorithm (module1)
        tm.update_missing_objects_tracker(frame)
        # Send current frame detections to Data Association module (module2)
        da.update(rects, scores, prev_frame=prev_frame, frame=frame)
        # For every detection on frame T, check their quality, if enrollable or verifiable obtain feature vector
        # and add it to the mean of features template of that tracklet (module3)
        fbtr.calculate_quality_all_dets(manager, frame)
        # ... for each tracklet Tk with an assigned detection Dk in the current frame, we retrieve tracklets Ti
        # with i != k. (module3)
        list_of_pairs = fbtr.compute_face_similarities(manager)
        # apply module 4
        cm.correct_pairs(list_of_pairs)
        # Tell tracklets to save their position on this frame
        manager.add_to_tracklets_histories(count)
        # Save a reference for the t-1 frame
        prev_frame = frame

        # If we want to see things happening, render video. This should be off during
        # actual benchmark
        if config_dict["show"]:  # config.SHOW
            draw_boxes_and_ids(manager.active_tracklets, frame)
            # Resize to fit on screen
            cv2.imshow("MOT17-09", image_resize(frame, height=600))
            _ = cv2.waitKey(1) & 0xFF
        # count frame number
        count += 1
    # do a bit of cleanup
    cv2.destroyAllWindows()
    # Write down results to txt file for metric measuring
    if config_dict["write_txt"]:
        write_results_mot_format("MOT17-09", manager.active_tracklets, manager.inactive_tracklets)
