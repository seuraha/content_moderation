import numpy as np
from numpy.random import permutation, multivariate_normal
import pandas as pd
import content_moderation_constants as C
from content_moderation_model import evaluation
from content_moderation_data_code import sample_user_data, sample_note_data
import json

class RatingSimulation:
    def __init__(self, configs, strategic_users: list=[]):
        self.configs, self.is_user_init, self.is_note_init = self.get_configs(configs)

        self.strategic_users = strategic_users

        self.user_set_I, self.user_set_I_R, self.user_set_I_L, self.user_id_map = self.get_user_set()
        self.note_set_J, self.note_id_map = self.get_note_set()

    def get_configs(self, configs: dict=None):
        try:
            last_configs = json.load(open(C.last_configs_path, "r"))
        except:
            assert configs is not None, "No configuration provided"
            last_configs = {k: None for k in configs.keys()}

        if configs:
            json.dump(configs, open(C.last_configs_path, "w"))
            
            is_user_init, is_note_init = False, False

            print("Updated configs:")
            for k in configs.keys():
                if last_configs[k] != configs[k]:
                    print(f"\t{k}: {last_configs[k]} -> {configs[k]}")
                    is_user_init = True if "user" in k else is_user_init
                    is_note_init = True if "note" in k else is_note_init
            if not is_user_init:
                print("\tuser configs unchanged")
            if not is_note_init:
                print("\tnote configs unchanged")

        else:
            configs = last_configs
        
        return configs, is_user_init, is_note_init

    
    def get_user_set(self):
        strategic_users_n = len(self.strategic_users)

        if self.is_user_init:
            from numpy.random import normal
            configs = self.configs

            user_n = configs["user_n"]

            n_R = int(user_n * configs["user_pi_R"])       # total number of users in group R
            n_L = user_n - n_R                             # total number of users in group L

            r_dim_value = normal(configs["user_opinion_R"][1], configs["user_opinion_R_std"], n_R).reshape((-1,1))
            r_dim_accuracy = np.sqrt(1 - r_dim_value**2)

            l_dim_value = normal(configs["user_opinion_L"][1], configs["user_opinion_L_std"], n_L).reshape((-1,1))
            l_dim_accuracy = np.sqrt(1 - l_dim_value**2)

            group_opinion_R = np.hstack((r_dim_accuracy, r_dim_value))
            group_opinion_L = np.hstack((l_dim_accuracy, l_dim_value))

            _ = np.arange(user_n).reshape(-1,1)               # user indices
        
            I_R = np.hstack(
                (_[:n_R], group_opinion_R)
                )   # set of users in group R: matrix of dimension `n_R` X 3
            I_L = np.hstack(
                (_[n_R:], group_opinion_L)
                )   # set of users in group L: matrix of dimension `n_L` X 3
            
            sampled_participantId, skiprows = sample_user_data(user_n)
            
            np.save(C.baseline_I_R_path, I_R)
            np.save(C.baseline_I_L_path, I_L)
            
            user_id_map = {"user_sim_ID": _.flatten().tolist(), "participantId": sampled_participantId, "index": skiprows}
            json.dump(user_id_map, open(C.user_id_map_path, "w"))

        else:
            I_R = np.load(C.baseline_I_R_path)
            I_L = np.load(C.baseline_I_L_path)

            user_id_map = json.load(open(C.user_id_map_path, "r"))

        if strategic_users_n > 0:
            I_strategic_R = np.array([[u.sim_id, u.opinion[0], u.opinion[1]] for u in self.strategic_users if u.group == "R"])
            I_strategic_L = np.array([[u.sim_id, u.opinion[0], u.opinion[1]] for u in self.strategic_users if u.group == "L"])

            I_R = np.vstack((I_strategic_R, I_R)) if len(I_strategic_R) > 0 else I_R
            I_L = np.vstack((I_strategic_L, I_L)) if len(I_strategic_L) > 0 else I_L

            user_id_map['user_sim_ID'].extend([u.sim_id for u in self.strategic_users])
            user_id_map['participantId'].extend([u.data_id for u in self.strategic_users])
            user_id_map['index'].extend([-1] * strategic_users_n)
        
        I = np.vstack((I_R, I_L)) # set of all users
            
        return I, I_R, I_L, user_id_map

    def get_note_set(self):
        if self.is_note_init:
            note_n = self.configs["note_n"]
            _ = np.array(range(note_n)).reshape(-1,1)    # note indices
            note_opinions = multivariate_normal(
                self.configs["note_mu"], 
                self.configs["note_cov"], 
                note_n)        # note opinion features: multivariate normal
            J = np.hstack((_, note_opinions)) # set of all notes: matrix of dimension `n_notes` X 3

            np.save(C.J_path, J)

            sampled_noteId, createdAtMillis = sample_note_data(note_n)
            note_id_map = {
                "note_sim_ID": _.flatten().tolist(), 
                "noteId": sampled_noteId,
                "createdAtMillis": createdAtMillis}
            json.dump(note_id_map, open(C.note_id_map_path, "w"))
        else:
            J = np.load(C.J_path)

            note_id_map = json.load(open(C.note_id_map_path, "r"))

        return J, note_id_map

    def assign_raters(self):
        rating_R_n, rating_L_n = self.configs["rating_R_n"], self.configs["rating_L_n"]
        notes = np.repeat(self.note_set_J, rating_R_n + rating_L_n, axis=0)
        assigned_raters = np.vstack(
            np.apply_along_axis(
            lambda x: np.vstack(
                (permutation(self.user_set_I_R)[:rating_R_n], permutation(self.user_set_I_L)[:rating_L_n])
                ),
            1, 
            self.note_set_J)
        )
        dot_product = notes[:,1] * assigned_raters[:,1] + notes[:,2] * assigned_raters[:,2]
        
        return pd.DataFrame(
            {"note_sim_ID": notes[:,0], 
             "note_dim_accuracy": notes[:,1], 
             "note_dim_value": notes[:,2], 
             "user_sim_ID": assigned_raters[:,0], 
             "user_dim_accuracy": assigned_raters[:,1],
             "user_dim_value": assigned_raters[:,2],
             "dot_product": dot_product,
             "evaluation": np.vectorize(evaluation)(dot_product)})
    