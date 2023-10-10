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
            opinion_value: float = 0.5,
            ):
        self.sim_id = sim_id
        self.data_id = self.sample_ID()
        self.group = group
        self.opinion = np.array([np.sqrt(1-opinion_value**2), opinion_value])

    def sample_ID(self):
        skiprows = json.load(open(C.user_id_map_path, "r"))['index']
        id = sample_user_data(1, skiprows=skiprows)
        return id