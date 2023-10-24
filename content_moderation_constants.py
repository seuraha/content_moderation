import os

data_path = "Data_upto_09152023_2031"

sourcecode_path = "communitynotes-main/sourcecode"
sourcecode_import_path = "communitynotes-main.sourcecode"

sim_data_path = os.path.join(sourcecode_path, "sim_data")
exp_data_path = os.path.join(sim_data_path, "exp_data")

# Input datasets paths
users_path = os.path.join(sim_data_path, "userEnrollment.tsv")
notes_path = os.path.join(sim_data_path, "notes.tsv")
ratings_path = os.path.join(sim_data_path, "ratings.tsv")
status_path = os.path.join(sim_data_path, "note_status_history.tsv")

# Simulated ID and model datasets path
sim_ratings_path = os.path.join(sim_data_path, "sim_ratings.tsv")
sim_users_ID_o_path = os.path.join(sim_data_path, "sim_users_ID_o.tsv")
sim_notes_ID_o_path = os.path.join(sim_data_path, "sim_notes_ID_o.tsv")

last_configs_path = os.path.join(sim_data_path, "last_configs.json")

baseline_I_R_path = os.path.join(sim_data_path, "sim_baseline_I_R.npy")
baseline_I_L_path = os.path.join(sim_data_path, "sim_baseline_I_L.npy")
user_id_map_path = os.path.join(sim_data_path, "user_id_map.json")

J_path = os.path.join(sim_data_path, "sim_note_J.npy")
note_id_map_path = os.path.join(sim_data_path, "note_id_map.json")

# Simulated ID and model datasets path Grid Experiment
grid_exp_data_path = os.path.join(sim_data_path, "grid_exp")

grid_last_configs_path = os.path.join(grid_exp_data_path, "last_configs.json")

grid_ratings_path = os.path.join(grid_exp_data_path, "grid_ratings.tsv")
grid_users_ID_o_path = os.path.join(grid_exp_data_path, "grid_users_ID_o.tsv")
grid_notes_ID_o_path = os.path.join(grid_exp_data_path, "grid_notes_ID_o.tsv")

grid_baseline_users_path = os.path.join(grid_exp_data_path, "grid_baseline_users.npy")
grid_J_path = os.path.join(grid_exp_data_path, "grid_note_J.npy")

grid_user_id_map_path = os.path.join(grid_exp_data_path, "grid_user_id_map.json")
grid_note_id_map_path = os.path.join(grid_exp_data_path, "grid_id_map.json")

# Matrix factorization results path
scored_notes_path = os.path.join(sim_data_path, "scored_notes.tsv")
helpfulness_scores_path = os.path.join(sim_data_path, "helpfulness_scores.tsv")
note_status_history_path = os.path.join(sim_data_path, "note_status_history.tsv")
aux_note_info_path = os.path.join(sim_data_path, "aux_note_info.tsv")

note_status_history_columns = [
    'noteId',
    'noteAuthorParticipantId',
    'createdAtMillis',
    'timestampMillisOfFirstNonNMRStatus',
    'firstNonNMRStatus',
    'timestampMillisOfCurrentStatus',
    'currentStatus',
    'timestampMillisOfLatestNonNMRStatus',
    'mostRecentNonNMRStatus',
    'timestampMillisOfStatusLock',
    'lockedStatus',
    'timestampMillisOfRetroLock']

verbose = True
