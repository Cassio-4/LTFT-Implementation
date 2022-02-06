arcface_config = {
    "model_path": "./resnet18_110.pth",
    "mode": "cuda",
    "threshold": 0.8,
    "batch_size": 10
}
_3ddfa_config = {
    "mode": "cuda",
    "bbox_init": "one",
    "path_3ddfa_model": "./3DDFA/models/phase1_wpdc_vdc.pth.tar",
    "recognition_threshold": 0.7
}
data_association_config = {
    "t_max": 20,
    "lambda_iou": 0.25
}
config_dict = {
    "arcface_config": arcface_config,
    "3ddfa_config": _3ddfa_config,
    "data_association_config": data_association_config,
    "show": False,
    "write_txt": True,
    "videos_folder": "./data/VideoDataset/",
    "videos": ["MOT17-09_video.avi", "MOT17-01_video.avi", "MOT17-04_video.avi", "Street_CutVideo.avi",
               "Sidewalk_CutVideo.avi", "Bengal_CutVideo.avi", "Terminal1_CutVideo.avi", "Terminal2_CutVideo.avi",
               "Terminal3_CutVideo.avi", "Terminal4_CutVideo.avi", "Shibuya_CutVideo.avi", "Choke1_CutVideo.avi",
               "Choke2_CutVideo.avi"],
    "dets_folder": "./data/detections/"
}

