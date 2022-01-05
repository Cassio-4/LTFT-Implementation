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


def get_all_detections_paths(dets_folder=""):
    dets_paths = []
    # Grab all sequence related frame paths
    detections_path = dets_folder + "/*.txt"
    for filename in glob.glob(detections_path):
        if "MOT17-09" in filename:
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
