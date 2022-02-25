from utils import image_resize, draw_boxes_and_ids, write_results_mot_format, get_detections_as_dictionary, count_verifiable_enrollable
from modules.fbtr import FaceBasedTrackletReconnectionModule
from modules.data_association import DataAssociationModule
from modules.TrackletManager import TrackletManager
from modules.tracking import TrackingModule
from modules.correction import CorrectionModule
from config import config_dict
import cv2


if __name__ == '__main__':
    fbtr = FaceBasedTrackletReconnectionModule(config_dict["fbtr_config"], config_dict["arcface_config"])

    for video in config_dict["videos"]:
        # For expediency, I am reading the detections from detection files instead of actually running the detector
        detections = get_detections_as_dictionary(config_dict["dets_folder"], video, config_dict["detection_threshold"])
        # Initialize every module
        manager = TrackletManager()
        tm = TrackingModule(tracklet_manager=manager)
        da = DataAssociationModule(config_dict["data_association_config"], tracklet_manager=manager)
        cm = CorrectionModule(manager)
        # Initialize frame counter, MOT17 videos start on frame 1
        count = 0
        if "MOT17" in video:
            count = 1
        prev_frame = None
        # Start pointer to video
        vc = cv2.VideoCapture(config_dict["videos_folder"] + video)
        # Read first frame
        success, frame = vc.read()
        while success:
            # Ready all detections on current frame
            rects = []
            scores = []
            dets = detections.pop(count, None)
            if dets is None:
                # If the model didn't detect any faces on this frame just print this, no problem tho
                print("video {} -> frame {} info doesnt exist".format(video, count))
                dets = []
            for det in dets:
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

            # If we want to see things happening, render video. This should be off during
            # actual benchmark
            if config_dict["show"]:  # config.SHOW
                draw_boxes_and_ids(manager.active_tracklets, frame)
                # Resize to fit on screen
                cv2.imshow(video, image_resize(frame, height=600))
                _ = cv2.waitKey(1) & 0xFF

            # Save a reference for the t-1 frame
            prev_frame = frame
            # Get next frame
            success, frame = vc.read()
            # count frame number
            count += 1
        # Cleanup at the end of sequence
        if config_dict["show"]:
            cv2.destroyAllWindows()
        # Write down results to txt file for metric measuring
        if config_dict["write_txt"]:
            write_results_mot_format(video.split("_")[0], manager.active_tracklets, manager.inactive_tracklets)
            count_verifiable_enrollable(manager.active_tracklets, manager.inactive_tracklets)
        # Write simple terminal ouput for peace of mind
        print("End of {}'s test".format(video))
        print("{} Frames processed".format(count))
