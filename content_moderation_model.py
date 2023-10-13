import numpy as np
import json
import content_moderation_constants as C
from content_moderation_data_code import sample_user_data

def evaluation(dot_product, thresholds=(0.5, 0)):
    threshold_h, threshold_s = thresholds
    return "HELPFUL" if dot_product >= threshold_h else "SOMEWHAT_HELPFUL" if dot_product >= threshold_s else "NOT_HELPFUL"

class StrategicUser:
    def __init__(
            self, 
            sim_id,
            group: str = "R",
            opinion_value: float = 1/2**(1/2),
            ):
        self.sim_id = sim_id
        self.data_id, self.data_index = self.get_data_id()
        self.group = group
        self.opinion = np.array([np.sqrt(1-opinion_value**2), opinion_value])

    def get_data_id(self):
        user_id_map = json.load(open(C.user_id_map_path, "r"))
        try:
            idx = user_id_map['user_sim_ID'].index(self.sim_id)
            return user_id_map['participantId'][idx], user_id_map['index'][idx]
        except:
            return None, None