import numpy as np
from numpy.random import permutation, multivariate_normal
import pandas as pd
import content_moderation_constants as C
from content_moderation_model import evaluation
from content_moderation_data_code import sample_user_data, sample_note_data
import json


class RatingGridSimulation:
    def __init__(self, configs, strategic_users: list = [], user_grid=True):
        self.configs = configs
        self.is_user_init = False
        self.is_note_init = False
        self.get_configs()

        self.note_set_J = None
        self.note_id_map = None
        self.initialize_notes()

        self.user_grid = user_grid
        self.strategic_users = strategic_users
        self.user_set_I = None
        self.user_id_map = None
        self.initialize_users()

    def get_configs(self):
        if self.configs["init_all"]:
            self.is_user_init = True
            self.is_note_init = True
        else:
            try:
                last_configs = json.load(open(C.grid_last_configs_path, "r"))
            except:
                assert self.configs is not None, "No configuration provided"
                last_configs = {k: None for k in self.configs.keys()}

            print("Updated configs:")
            for k in self.configs.keys():
                if last_configs[k] != self.configs[k]:
                    print(f"\t {k}: {last_configs[k]} -> {self.configs[k]}")
                    self.is_user_init = True if "user" in k else self.is_user_init
                    self.is_note_init = True if "note" in k else self.is_note_init

            if not self.is_user_init:
                print("\t user configs unchanged")
            if not self.is_note_init:
                print("\t note configs unchanged")

        json.dump(self.configs, open(C.grid_last_configs_path, "w"))

    def initialize_users(self):
        if self.is_user_init:
            user_n = self.configs["user_n"]
            ids = np.arange(user_n).reshape(-1, 1)

            n_R = int(user_n * self.configs["user_pi_R"])
            n_L = user_n - n_R

            if self.user_grid:
                angles_R = np.linspace(1 / 2 * np.pi, 0, n_R)
                angles_L = np.linspace(1 / 2 * np.pi, 0, n_L)
            else:
                angles_R = np.random.normal((1 / 4) * np.pi, 0.2, n_R)
                angles_L = np.random.normal((1 / 4) * np.pi, 0.2, n_L)

            r_dim_value, r_dim_accuracy = np.cos(angles_R), np.sin(angles_R)
            l_dim_value, l_dim_accuracy = -np.cos(angles_L), np.sin(angles_L)

            group_opinion_R = np.hstack(
                (r_dim_accuracy.reshape((-1, 1)), r_dim_value.reshape((-1, 1)))
            )
            group_opinion_L = np.hstack(
                (l_dim_accuracy.reshape((-1, 1)), l_dim_value.reshape((-1, 1)))
            )
            opinions = np.vstack((group_opinion_R, group_opinion_L))
            I = np.hstack((ids, opinions))

            try:
                sampled_participantId = np.load(C.grid_baseline_users_path)
            except:
                sampled_participantId = sample_user_data(user_n)
            np.save(C.grid_baseline_users_path, I)

            user_id_map = {
                "user_sim_ID": ids.flatten().tolist(),
                "participantId": sampled_participantId,
            }
            json.dump(user_id_map, open(C.grid_user_id_map_path, "w"))

        else:
            I = np.load(C.grid_baseline_users_path)
            user_id_map = json.load(open(C.grid_user_id_map_path, "r"))

        if len(self.strategic_users) > 0:
            need_data_id_u = [
                u
                for u in self.strategic_users
                if u.sim_id not in user_id_map["user_sim_ID"]
            ]
            need_data_id_n = len(need_data_id_u)

            if need_data_id_n > 0:
                sampled_participantId = sample_user_data(
                    need_data_id_n, drop_ids=user_id_map["participantId"]
                )

                for i, u in enumerate(need_data_id_u):
                    u.data_id = sampled_participantId[i]
                    user_id_map["user_sim_ID"].append(u.sim_id)
                    user_id_map["participantId"].append(u.data_id)

                json.dump(user_id_map, open(C.grid_user_id_map_path, "w"))

            I_strategic = np.array(
                [[u.sim_id, u.opinion[0], u.opinion[1]] for u in self.strategic_users]
            )

            I = np.vstack((I_strategic, I))  # set of all users

        return I, user_id_map

    def note_id_mapper(self, sim_id=None, data_id=None):
        id_map = pd.DataFrame(self.note_id_map)
        if sim_id:
            c = id_map.loc[:, "note_sim_ID"] == sim_id
            return id_map.loc[c, "noteId"]
        if data_id:
            c = id_map.loc[:, "noteId"] == data_id
            return id_map.loc[c, "note_sim_ID"]

    def initialize_notes(self):
        if self.is_note_init:
            note_grid_n = self.configs["note_grid_n"]
            note_n = note_grid_n**2
            x = np.linspace(-1, 1, note_grid_n)
            xg, yg = np.meshgrid(x, x)

            ids = np.arange(note_n).reshape(-1, 1)  # note indices
            note_opinions = np.hstack((xg.reshape(-1, 1), yg.reshape(-1, 1)))
            J = np.hstack(
                (ids, note_opinions)
            )  # set of all notes: matrix of dimension `n_notes` X 3

            np.save(C.grid_J_path, J)

            sampled_noteId, createdAtMillis = sample_note_data(note_n)
            note_id_map = {
                "note_sim_ID": ids.flatten().tolist(),
                "noteId": sampled_noteId,
                "createdAtMillis": createdAtMillis,
            }
            json.dump(note_id_map, open(C.grid_note_id_map_path, "w"))
        else:
            J = np.load(C.grid_J_path)
            note_id_map = json.load(open(C.grid_note_id_map_path, "r"))

        return J, note_id_map

    def assign_random_raters_from_group(self, rating_R_n, rating_L_n):
        notes = np.repeat(self.note_set_J, rating_R_n + rating_L_n, axis=0)
        assigned_raters = np.vstack(
            np.apply_along_axis(
                lambda x: np.vstack(
                    (
                        permutation(self.user_set_I_R)[:rating_R_n],
                        permutation(self.user_set_I_L)[:rating_L_n],
                    )
                ),
                1,
                self.note_set_J,
            )
        )
        dot_product = (
            notes[:, 1] * assigned_raters[:, 1] + notes[:, 2] * assigned_raters[:, 2]
        )

        return pd.DataFrame(
            {
                "note_sim_ID": notes[:, 0],
                "note_dim_accuracy": notes[:, 1],
                "note_dim_value": notes[:, 2],
                "user_sim_ID": assigned_raters[:, 0],
                "user_dim_accuracy": assigned_raters[:, 1],
                "user_dim_value": assigned_raters[:, 2],
                "user_group": np.where(assigned_raters[:, 2] > 0, "R", "L"),
                "dot_product": dot_product,
                "evaluation": np.vectorize(evaluation, otypes=[float])(dot_product),
            }
        )

    def assign_raters_frac(self, dim_frac=1, strategic_user_grid=True):
        np.random.seed(0)

        if strategic_user_grid:
            strategic_user = self.user_set_I[0]
            other_users = self.user_set_I[1:]
            rater_n = int(len(other_users) * (dim_frac**2))

            coarser_grid_step = int((dim_frac) ** (-1))
            x = np.linspace(-1, 1, self.configs["note_grid_n"])[::coarser_grid_step]
            note_df = pd.DataFrame(self.note_set_J)

            strategic_user_notes = note_df.loc[
                (note_df[1].isin(x)) & (note_df[2].isin(x))
            ].to_numpy()
            notes = np.repeat(self.note_set_J, rater_n, axis=0)
            assigned_raters = np.vstack(
                np.apply_along_axis(
                    lambda x: np.vstack((permutation(other_users)[:rater_n])),
                    1,
                    self.note_set_J,
                )
            )
            notes = np.vstack((strategic_user_notes, notes))
            assigned_raters = np.vstack(
                (
                    np.repeat([strategic_user], len(strategic_user_notes), axis=0),
                    assigned_raters,
                )
            )
        else:
            rater_n = int(self.configs["user_n"] * dim_frac)
            notes = np.repeat(self.note_set_J, rater_n, axis=0)
            assigned_raters = np.vstack(
                np.apply_along_axis(
                    lambda x: np.vstack((permutation(self.user_set_I)[:rater_n])),
                    1,
                    self.note_set_J,
                )
            )
        dot_product = (
            notes[:, 1] * assigned_raters[:, 1] + notes[:, 2] * assigned_raters[:, 2]
        )

        return pd.DataFrame(
            {
                "note_sim_ID": notes[:, 0],
                "note_dim_accuracy": notes[:, 1],
                "note_dim_value": notes[:, 2],
                "user_sim_ID": assigned_raters[:, 0],
                "user_dim_accuracy": assigned_raters[:, 1],
                "user_dim_value": assigned_raters[:, 2],
                "user_group": np.where(assigned_raters[:, 2] > 0, "R", "L"),
                "dot_product": dot_product,
                "evaluation": np.vectorize(evaluation, otypes=[float])(dot_product),
            }
        )
