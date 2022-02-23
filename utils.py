import glob
import cv2
import os

from calcs import centroid_from_bbox


def get_all_frames_path(frames_folder="/home/cassio/dataset/Images/MOT17-09"):
    img_paths = []
    # Grab all sequence related frame paths
    sequence_path = frames_folder + "/*.jpg"
    for filename in glob.glob(sequence_path):
        img_paths.append(filename)
    # Sort all paths
    img_paths = sorted(img_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    return img_paths


def get_all_detections_paths(dets_folder, video_name):
    dets_paths = []
    # Grab all sequence related frame paths
    detections_path = dets_folder + "/*.txt"
    for filename in glob.glob(detections_path):
        if video_name.split("_")[0].lower() in filename.lower():
            dets_paths.append(filename)
    # Sort all paths
    dets_paths = sorted(dets_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].split("_")[1]))
    return dets_paths


def get_framedets_asarray(det_file, thresh):
    framedets = []
    with open(det_file, 'r') as f:
        lines = f.readlines()
        # remove \n
        lines = [line.rstrip() for line in lines]
        for line in lines:
            line = line.split(" ")
            # face 0.85 717 397 782 477
            score = float(line[1])
            xmin = int(line[2])
            ymin = int(line[3])
            xmax = int(line[4])
            ymax = int(line[5])
            if score >= thresh:
                framedets.append((score, xmin, ymin, xmax, ymax))
    return framedets


def draw_boxes_and_ids(objects, frame):
    # loop over the tracked objects
    for (objectID, tklet) in objects.items():
        # Paint it green if the tracklet is not missing
        if tklet.active:
            color = (0, 255, 0)
        # Paint it red if missing
        else:
            color = (0, 0, 255)
        # id number
        text = "{}".format(objectID)
        # get center of bbox
        centroid = centroid_from_bbox(tklet.position)
        # Draw object's id on image
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)
        # Draw bbox around the object (face)
        pos = tklet.position
        frame = cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), color, 2)
    return frame


def get_detections_as_dictionary(path_to_files, video_name, thresh=0.75):
    dets_paths = get_all_detections_paths(path_to_files, video_name)
    det_dict = {}
    for f in dets_paths:
        number = f.split("/")[-1].split("_")[-1].split(".")[0]
        number = int(number)
        dets = get_framedets_asarray(f, thresh)
        det_dict[number] = dets
    return det_dict


def ready_frames_and_detections(frames_folder="/home/cassio/dataset/Images/MOT17-09", dets_folder="", thresh=0.75):
    img_paths = get_all_frames_path(frames_folder)
    dets_paths = get_all_detections_paths(dets_folder)
    # This is dumb
    list_of_tuples = []
    for img, det_path in zip(img_paths, dets_paths):
        dets = get_framedets_asarray(det_path, thresh)
        list_of_tuples.append((img, dets))
    return list_of_tuples


def draw_boxes_on_image(image, boxes):
    """
    Draws bounding boxes and Ids on image
    :param boxes: list of tuples (score, xmin, ymin, xmax, ymax)
    :param image: the image to draw on (using opencv)
    :return: the drawn on frame
    """
    image_copy = image.copy() #cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    color = (0, 255, 0)
    for box in boxes:
        # (score, xmin, ymin, xmax, ymax)
        cv2.rectangle(image_copy, (box[1], box[2]), (box[3], box[4]), color, 2)
    return image_copy


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def write_results_mot_format(video_name, active_tracklets_dict, inactive_tracklets_dict):
    # Build a dictionary that maps information to frames
    # [int:frame number, list of tuples[(id, xmin, ymin, xmax, ymax),...]]
    frames_dict = {}
    iterate_history_dicts(frames_dict, active_tracklets_dict)
    iterate_history_dicts(frames_dict, inactive_tracklets_dict)
    line_count = 0
    frame_count = 0
    with open("./data/results/{}_mot.txt".format(video_name), 'w') as f:
        while frames_dict:
            try:
                frame_info = frames_dict.pop(frame_count)
            except KeyError:
                # This just means that the frame #frame_count has no annotations if every info
                # has been extracted from the dictionary the while will stop when false
                frame_count += 1
                continue
            # frame_info is a list of tuples, each tuple an annotation (id, xmin, ymin, xmax, ymax)
            for tup in frame_info:
                t_id = int(tup[0])
                xmin = int(tup[1])
                ymin = int(tup[2])
                xmax = int(tup[3])
                ymax = int(tup[4])
                bb_width = int(xmax - xmin)
                bb_height = int(ymax - ymin)
                # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                line = "{}, {}, {}, {}, {}, {}, -1, -1, -1, -1".format(frame_count, t_id, xmin, ymin, bb_width,
                                                                       bb_height)
                # if first line then we write without the \n at the beginning
                if line_count == 0:
                    f.write(line)
                else:
                    f.write("\n{}".format(line))
                line_count += 1
            frame_count += 1


def iterate_history_dicts(frames_dict, history_dict):
    for tracklet in history_dict.values():
        t_id = tracklet.id
        for frame_num, position in tracklet.position_history.items():
            # If this frame doesnt exist in the dictionary, instantiate an empty list for it
            if not (frame_num in frames_dict):
                frames_dict[frame_num] = []
            #           (id,   xmin,        ymin,        xmax,        ymax)
            pos_tuple = (t_id, position[0], position[1], position[2], position[3])
            frames_dict[frame_num].append(pos_tuple)
