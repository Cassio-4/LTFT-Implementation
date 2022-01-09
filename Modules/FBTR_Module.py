import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import scipy.io as sio
import numpy as np
import torch
import cv2
parse_roi_box_from_landmark = __import__('3DDFA.utils.inference.parse_roi_box_from_landmark')
parse_roi_box_from_bbox = __import__('3DDFA.utils.inference.parse_roi_box_from_bbox')
predict_68pts = __import__('3DDFA.utils.inference.predict_68pts')
parse_pose = __import__('3DDFA.utils.estimate_pose.parse_pose')
NormalizeGjz = __import__('3DDFA.utils.ddfa.NormalizeGjz')
ToTensorGjz = __import__('3DDFA.utils.ddfa.ToTensorGjz')
crop_img = __import__('3DDFA.utils.inference.crop_img')
str2bool = __import__('3DDFA.utils.ddfa.str2bool')
mobilenet_v1 = __import__('3DDFA.mobilenet_v1')



class FBTR_Module:
    def __init__(self, recognition_threshold=0.5, mode='cpu', bbox_init='one',
                 path_3ddfa_model="../3DDFA/models/phase1_wpdc_vdc.pth.tar"):
        self.lambda_recognition = recognition_threshold
        self.mode = mode
        self.STD_SIZE = 120
        self.bbox_init = bbox_init
        print("Loading 3DDFA pose estimation model")
        self.model_3ddfa = self.load_3ddfa_model(path_3ddfa_model)
        print("Done loading 3DDFA model.")

    # Implementation of section 2.2 of https://gc2011.graphicon.ru/html/2014/papers/111-114.pdf
    def get_blur_metric(self, face_image, landmarks):
        # Get an even smaller bbox from the landmark points
        rect = cv2.boundingRect(landmarks)
        x, y, w, h = rect
        # Crop the face from the image using the landmark points bbox
        cropped = face_image[y:y + h, x:x + w].copy()
        # Build a mask to remove background information
        pts = landmarks - landmarks.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # Use mask to crop face using bitwise and operation
        final = cv2.bitwise_and(cropped, cropped, mask=mask)
        # Use Laplace filter to obtain contours and edges, the more sharp an image is the greater the response
        # from the laplace filter
        lap = cv2.Laplacian(final, cv2.CV_32F)
        # The paper where this idea originated from says: "The image
        # sharpness score is calculated as the averaged Laplace
        # operator response by masked image".
        blur_score = np.average(lap)
        return blur_score

    def forward_3ddfa(self, face_images):
        """
        Pass image through the 3DDFA model to get pose and landmark points
        :param face_images: A cropped face detection
        """

        tri = sio.loadmat('visualize/tri.mat')['tri']
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

        for img_fp in face_images:
            # This pipeline doesn't need to load image from file, we're already providing face detections
            img_ori = img_fp
            # Initiating placeholders
            pts_res = []
            # Ps = []  # Camera matrix collection

            vertices_lst = []  # store multiple face vertices
            ind = 0
            # This line gets the file suffix to save the results using a similar name, there's no need for this
            # suffix = get_suffix(img_fp)

            # This would be the crop of the face, however our face images should already arrive cropped
            img = img_ori  # crop_img(img_ori, roi_box)
            # Since we are using already cropped images, our roi box is the dimensions of the picture
            layer_num, row_num, col_num = img.shape
            bbox = [0, 0, row_num-1, col_num-1]#[rect.left(), rect.top(), rect.right(), rect.bottom()]
            roi_box = parse_roi_box_from_bbox(bbox)
            # forward: one step
            img = cv2.resize(img, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if self.mode == 'gpu':
                    input = input.cuda()
                param = self.model_3ddfa(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if self.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if self.mode == 'gpu':
                        input = input.cuda()
                    param = self.model_3ddfa(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)
            return pose, pts68

    def get_score(self, any):
        pass

    def load_3ddfa_model(self, path):
        checkpoint_fp = path
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        if self.mode == "gpu":
            cudnn.benchmark = True
            model = model.cuda()
        model.eval()
        return model
