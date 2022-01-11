from utils import ready_frames_and_detections, image_resize, draw_boxes_and_ids
from Modules.DataAssociationModule import DataAssociationModule
from Modules.TrackletManager import TrackletManager
from Modules.TrackingModule import TrackingModule
from Modules.FBTR_Module import FBTR_Module
import cv2


if __name__ == '__main__':
    # Ready everything needed
    all_frames_and_dets = ready_frames_and_detections(frames_folder="/home/cassio/dataset/Images/MOT17-09",
                                                      dets_folder="/home/cassio/CrowdedDataset/Yolov5s-nothresh-clean"
                                                                  "-scoxyXY")
    # Initialize Tracklet Manager
    manager = TrackletManager()
    tm = TrackingModule(tracklet_manager=manager)
    da = DataAssociationModule(t_max=20, lambda_iou=0.1, tracklet_manager=manager)
    fbtr = FBTR_Module(mode='cpu')

    # main loop
    count = 0
    prev_frame = None
    for img_path, detections in all_frames_and_dets:
        # Read frame
        frame = cv2.imread(img_path)
        # Ready all detections on current frame
        rects = []
        for det in detections:
            rects.append(det[1:])
        # ========== Tracking Pipeline ==========
        # First update the positions for the objects that haven't been previously matched
        # thus are being tracked via specific tracking algorithm (module1)
        tm.update_missing_objects_tracker(frame)
        # Send current frame detections to Data Association module (module2)
        da.update(rects, prev_frame=prev_frame, frame=frame)
        # apply module 3
        # TODO call fbtr module
        # apply module 4
        # Save a reference for the t-1 frame
        prev_frame = frame

        # If we want to see things happening, render video. This should be off during
        # actual benchmark
        if True:  # config.SHOW
            draw_boxes_and_ids(manager.active_tracklets, frame)
            # Resize to fit on screen
            cv2.imshow("MOT17-09", image_resize(frame, height=600))
            _ = cv2.waitKey(1) & 0xFF

        count += 1
    # do a bit of cleanup
    cv2.destroyAllWindows()
