arcface_config = {
    "model_path": "./resnet18_110.pth",
    "mode": "cpu",
    "threshold": 0.8,
    "batch_size": 10
}
_3ddfa_config = {
    "mode": "cpu",
    "bbox_init": "one",
    "path_3ddfa_model": "./3DDFA/models/phase1_wpdc_vdc.pth.tar",
    "recognition_threshold": 0.7
}
data_association_config = {
    "t_max": 20,
    "lambda_iou": 0.1
}
config_dict = {
    "arcface_config": arcface_config,
    "3ddfa_config": _3ddfa_config,
    "data_association_config": data_association_config,
    "show": True
}
