arcface_config = {
    "model_path": "./resnet18_110.pth",
    "mode": "cuda",  # "cpu" or "cuda"
    "batch_size": 10  # there's no batch processing right now, add later
}
fbtr_config = {
    "3ddfa_mode": "gpu",  # "cpu" or "gpu"
    "3ddfa_bbox_init": "one",
    "3ddfa_model_path": "./3DDFA/models/phase1_wpdc_vdc.pth.tar",
    "recognition_threshold": 0.5,
    "blur_thresholds": (0.9, 0.75),  # Upper and lower blur thresholds respectively
    "e_margin": 0.8,
    "C": 6
}
data_association_config = {
    "t_max": 20,
    "lambda_iou": 0.25
}
config_dict = {
    "arcface_config": arcface_config,
    "fbtr_config": fbtr_config,
    "data_association_config": data_association_config,
    "detection_threshold": 0.75,
    "show": False,
    "write_txt": True,
    "videos_folder": "./data/VideoDataset/",
    "videos": ["MOT17-09_video.avi", "MOT17-01_video.avi", "MOT17-04_video.avi", "Street_CutVideo.avi",
               "Sidewalk_CutVideo.avi", "Bengal_CutVideo.avi", "Terminal1_CutVideo.avi", "Terminal2_CutVideo.avi",
               "Terminal3_CutVideo.avi", "Terminal4_CutVideo.avi", "Shibuya_CutVideo.avi", "Choke1_CutVideo.avi",
               "Choke2_CutVideo.avi"],
    "dets_folder": "./data/detections/"
}

