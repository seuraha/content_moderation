import content_moderation_constants as C
import os
import pandas as pd
import numpy as np
import json

def get_ratings_data(
        sample_frac = 0.05
        ):

    data_ratings = pd.read_csv(os.path.join(C.data_path, "ratings-00000.tsv"), sep='\t', skiprows=lambda x: x > 0 and np.random.random() > sample_frac)
    try:
        data_ratings1 = pd.read_csv(os.path.join(C.data_path, "ratings-00001.tsv"), sep='\t', skiprows=lambda x: x > 0 and np.random.random() > sample_frac)
        data_ratings = pd.concat([data_ratings, data_ratings1]).reset_index(drop=True)

        del data_ratings1
    except:
        pass

    c = data_ratings["createdAtMillis"] > 1652918400000 # May 19, 2022
    data_ratings = data_ratings[c]

    return data_ratings.drop_duplicates()

def sample_user_data(
        sample_n,
        skiprows=None,
        data_path=C.data_path, 
        random_seed=0):

    data_user_enrollment = pd.read_csv(os.path.join(data_path, "userEnrollment-00000.tsv"), sep='\t', skiprows=skiprows)
    c = (data_user_enrollment["enrollmentState"]=="newUser") & (data_user_enrollment["modelingPopulation"]=="CORE") & (data_user_enrollment["modelingGroup"]==0)
    data_user_enrollment = data_user_enrollment[c].drop_duplicates()

    user_data_features = data_user_enrollment.sample(n=sample_n, random_state=random_seed)
    pid, idx = user_data_features["participantId"].to_list(), user_data_features.index.to_list()

    try:
        u = pd.read_csv(C.users_path, sep="\t")
        user_data_features = pd.concat([u, user_data_features])
        user_data_features.to_csv(C.users_path, sep="\t", index=False)
    
    except:
        user_data_features.to_csv(C.users_path, sep="\t", index=False)

    return pid, idx


def sample_note_data(sample_n, data_path=C.data_path, random_seed = 0):
    data_notes = pd.read_csv(os.path.join(data_path, "notes-00000.tsv"), sep='\t').drop_duplicates()
    note_features = data_notes.sample(n=sample_n, random_state=random_seed).reset_index(drop=True)

    note_features.to_csv(C.notes_path, sep="\t", index=False)
    note_status_data_to_tsv(note_features)
    return note_features["noteId"].to_list(), note_features["createdAtMillis"].to_list()

def note_status_data_to_tsv(note_features):
    import time
    note_status = pd.DataFrame(columns=C.note_status_history_columns)
    note_status["noteId"] = note_features["noteId"]
    note_status["noteAuthorParticipantId"] = note_features["noteAuthorParticipantId"]
    note_status["createdAtMillis"] = note_features["createdAtMillis"]
    note_status["timestampMillisOfCurrentStatus"] = round(time.time()*1000) # current time
    note_status["timestampMillisOfFirstNonNMRStatus"] = -1
    note_status["currentStatus"] = "NEEDS_MORE_RATINGS"
    note_status["timestampMillisOfLatestNonNMRStatus"] = -1
    note_status["timestampMillisOfRetroLock"] = -1
    note_status.to_csv(C.status_path, sep="\t", index=False)

def generate_ratings_data(
        simulated_ratings, 
        SimModel,
        random_seed=0,
        verbose=C.verbose):
    from numpy.random import binomial, exponential
    np.random.seed(random_seed)
    """
    In addition to helpfulness, Community Notes maintains tags to explain why users chose such a rating.
        e.g.,   if rated helpful, whether it was because it was clear, informative, or provides important contexts
                if rated not helpful, whether it was because it was outdated, unclear, or off topic.
                
    Wherever the rating is "helpful," simulates tags indicating reasons for rating helpful (binary);
    and wherever the rating is "not helpful," simulates tags indicating reasons for rating not helpful (binary);
    0, elsewhere.
    """
    avg_wait_time = 2*60*60*1000 # a note gets a rating once every 2 hours - this is not essential but to add "created at millisec" column to the rating data.
    if verbose:
        print("Loading ratings and id map data...")
    ratings_data = get_ratings_data()
    
    note_id_map_df = pd.DataFrame(SimModel.note_id_map)
    user_id_map_df = pd.DataFrame(SimModel.user_id_map)

    _ = simulated_ratings\
        .merge(note_id_map_df, on="note_sim_ID", how="left")\
        .merge(user_id_map_df, on="user_sim_ID", how="left")
    _["createdAtMillis"] = _["createdAtMillis"].apply(lambda x: x+int(exponential(scale=avg_wait_time, size=1)[0]))

    sim_ratings_df = pd.DataFrame(columns=ratings_data.columns)
    sim_ratings_df["noteId"] = _["noteId"]
    sim_ratings_df["raterParticipantId"] = _["participantId"]
    sim_ratings_df["version"] = 2
    sim_ratings_df["agree"] = 0       # deprecated, set to 0 by default in version 2
    sim_ratings_df["disagree"] = 0    # deprecated, set to 0 by default in version 2
    sim_ratings_df["helpful"] = 0     # deprecated, set to 0 by default in version 2
    sim_ratings_df["notHelpful"] = 0  # deprecated, set to 0 by default in version 2
    sim_ratings_df["helpfulnessLevel"] = _["rating"]
    sim_ratings_df["createdAtMillis"] = _["createdAtMillis"]
    sim_ratings_df["ratedOnTweetId"] = -1

    if verbose:
        print("Sampling explanation tags...")
    helpful_tags = [c for c in ratings_data.columns if c.startswith("helpful") and c not in ["helpful", "helpfulnessLevel"]]
    nothelpful_tags = [c for c in ratings_data.columns if c.startswith("notHelpful") and c not in ["notHelpful"]]

    # Indices
    data_helpful_idx = ratings_data["helpfulnessLevel"] == "HELPFUL"
    data_nothelpful_idx = ratings_data["helpfulnessLevel"] == "NOT_HELPFUL"

    sim_helpful_idx = sim_ratings_df["helpfulnessLevel"] == "HELPFUL"
    sim_nothelpful_idx = sim_ratings_df["helpfulnessLevel"] == "NOT_HELPFUL"

    # Create DataFrame with zeros
    sim_zeros = np.zeros((sim_ratings_df.shape[0], len(helpful_tags + nothelpful_tags)))
    sim_tag_df = pd.DataFrame(sim_zeros, columns = helpful_tags + nothelpful_tags)

    # For each tag, generate binomial random variables
    for c in helpful_tags:
        n = sum(sim_helpful_idx)
        mean = ratings_data.loc[data_helpful_idx, c].mean()
        sim_tag_df.loc[sim_helpful_idx, c] = binomial(1, mean, n)

    for c in nothelpful_tags:
        n = sum(sim_nothelpful_idx)
        mean = ratings_data.loc[data_nothelpful_idx, c].mean()
        sim_tag_df.loc[sim_nothelpful_idx, c] = binomial(1, mean, n)

    sim_ratings_df.fillna(sim_tag_df).to_csv(C.ratings_path, sep="\t", index=False)


def run_main(pseudoraters = True, 
             enabledScorers = None, 
             strictColumns = True, 
             runParallel = False):

    scoring_dir = __import__(C.sourcecode_import_path+".scoring", fromlist=["process_data", "run_scoring"])

    process_data = scoring_dir.process_data
    LocalDataLoader = process_data.LocalDataLoader
    write_tsv_local = process_data.write_tsv_local

    run_scoring = scoring_dir.run_scoring
    run_scoring = run_scoring.run_scoring

    """
    Just what their main() function does...
    """
    # Load input dataframes.
    dataLoader = LocalDataLoader(C.notes_path, C.ratings_path, C.status_path, C.users_path, True)
    _, ratings, statusHistory, userEnrollment = dataLoader.get_data()

    # Invoke scoring and user contribution algorithms.
    scoredNotes, helpfulnessScores, newStatus, auxNoteInfo = run_scoring(
        ratings,
        statusHistory,
        userEnrollment,
        seed=0,
        pseudoraters=pseudoraters,
        enabledScorers=enabledScorers,
        strictColumns=strictColumns,
        runParallel=runParallel,
        dataLoader=dataLoader if runParallel == True else None,
    )

    # Write outputs to local disk.
    write_tsv_local(scoredNotes, C.scored_notes_path)
    write_tsv_local(helpfulnessScores, C.helpfulness_scores_path)
    write_tsv_local(newStatus, C.note_status_history_path)
    write_tsv_local(auxNoteInfo, C.aux_note_info_path)
