"""
This script is a main wrapper, it is used to run multiple tests at once and save their results.
The parameters are set as ranges, and for every test a value is randomly chosen within that range.
Except for t_max, which is set to 5, because that seems the most viable based on empirical tests prior to this.
"""
from utils import image_resize, draw_boxes_and_ids, write_results_mot_format, get_detections_as_dictionary, count_verifiable_enrollable
from modules.fbtr import FaceBasedTrackletReconnectionModule
from modules.data_association import DataAssociationModule
from modules.TrackletManager import TrackletManager
from modules.tracking import TrackingModule
from modules.correction import CorrectionModule
from config import config_dict
import random, os, pickle
import cv2
# Number of random tests to be run
Ntests = 100
# Set of tuples to keep track of which parameters combinations have already been run
file_exists = os.path.exists("./combinations_set0-24.pickle")
if file_exists:
    with open("./combinations_set0-24.pickle", "rb") as input_file:
        Combinations_set = pickle.load(input_file)
    print("Loaded combinations file.")
else:
    Combinations_set = set()


def generate_random():
    # We assume that this set of parameters is not unique
    unique = False
    # While the generated set is not unique, generate a new set
    while not unique:
        # generate general detection threshold
        detection_threshold = round(random.uniform(0.65, 0.8), 2)
        # generate upper detection threshold
        upper_detection_threshold = round(random.uniform(detection_threshold+0.05, 0.9), 2)
        # generate lower detection threshold
        lower_detection_threshold = round(random.uniform(detection_threshold, upper_detection_threshold-0.01), 2)
        # generate upper blur threshold
        upper_blur_threshold = round(random.uniform(0.05, 3.0), 2)
        # generate lower blur threshold
        lower_blur_threshold = round(random.uniform(0.01, upper_blur_threshold-0.01), 2)
        # Create a tuple with these parameters
        param_tuple = (detection_threshold, upper_detection_threshold, lower_detection_threshold, upper_blur_threshold,
                       lower_blur_threshold)
        if param_tuple in Combinations_set:
            print("Combination {} {} {} {} {} already executed.".format(param_tuple[0], param_tuple[1], param_tuple[2],
                                                                        param_tuple[3], param_tuple[4]))
        else:
            Combinations_set.add(param_tuple)
            unique = True
    # Replace parameters on configuration dictionary (there's probably a smarter way to do this :p)
    config_dict["fbtr_config"]["blur_thresholds"] = (upper_blur_threshold, lower_blur_threshold)
    config_dict["fbtr_config"]["fbtr_det_score"] = (upper_detection_threshold, lower_detection_threshold)
    config_dict["detection_threshold"] = detection_threshold


if __name__ == '__main__':
    for test_number in range(Ntests):
        # Generate random parameters
        generate_random()
        # Create directory to save this test's files
        results_dir = "./data/results/yolov5s-random-{}".format(test_number)
        os.mkdir(results_dir)

        # Start test
        sum_of_verifiables, sum_of_enrollables = 0, 0
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
                    # If the model didn't detect any faces on this frame just print this
                    #print("video {} -> frame {} info doesnt exist".format(video, count))
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
                write_results_mot_format(video.split("_")[0], manager.active_tracklets, manager.inactive_tracklets, results_dir)
                verif, enrol = count_verifiable_enrollable(manager.active_tracklets, manager.inactive_tracklets)
                # add to counter of verifiable and enrollable faces
                sum_of_verifiables, sum_of_enrollables = sum_of_verifiables+verif, sum_of_enrollables+enrol
            # Write simple terminal ouput for peace of mind
            print("End of {}'s test".format(video))
        # At the end of this test, save total verifiables, total enrollables and test parameters in csv (;) files
        with open('quality_history.csv', 'a') as f:
            f.write("{};{};{}\n".format(test_number, sum_of_verifiables, sum_of_enrollables))
        with open('parameters history.csv', 'a') as p:
            upper_blur, lower_blur = config_dict["fbtr_config"]["blur_thresholds"]
            upper_det, lower_det = config_dict["fbtr_config"]["fbtr_det_score"]
            p.write("{};{};{};{};{};{};{}\n".format(test_number, config_dict["detection_threshold"], upper_det, lower_det,
                                                    upper_blur, lower_blur, config_dict["data_association_config"]["t_max"]))
        print("End of test {}.".format(test_number))

    # At the end of every test, dump the set as a pickle object in case we want to start from here (checkpoint)
    # Store data (serialize)
    print("Dumping parameters set on combinations_set.pickle file.")
    with open('combinations_set.pickle', 'wb') as handle:
        pickle.dump(Combinations_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("combinations_set.pickle dumped.")
