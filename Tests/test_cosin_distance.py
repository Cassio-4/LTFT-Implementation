from modules.TrackletManager import TrackletManager
from modules.fbtr import FaceBasedTrackletReconnectionModule
from config import config_dict
import numpy as np

fbtr = FaceBasedTrackletReconnectionModule(config_dict["3ddfa_config"], config_dict["arcface_config"])
manager = TrackletManager()
for i in range(6):
    manager.register([0]*4, 0.1)
manager.active_tracklets[0].active = False
manager.active_tracklets[1].active = False
manager.active_tracklets[2].active = False
for i in range(6):
    manager.active_tracklets[i].update_mean_enrollable(np.random.rand(1024))
    manager.active_tracklets[i].update_mean_verifiable(np.random.rand(1024))

fbtr.compute_face_similarities(manager)