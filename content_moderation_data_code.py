import content_moderation_constants as C
import os
import pandas as pd
import numpy as np
import json
from content_moderation_assessment import factorization_results

def get_ratings_data(
        sample_frac = 0.01
        ):
    deletedNoteTombstonesLaunchTime = 1652918400000  # May 19, 2022 UTC
    notMisleadingUILaunchTime = 1664755200000  # October 3, 2022 UTC
    data_ratings = pd.read_parquet(
        os.path.join(C.data_path, "ratings-00000.parquet"), 
        engine="pyarrow", 
        filters=[
            [('createdAtMillis', '>', deletedNoteTombstonesLaunchTime), ('createdAtMillis', '>', notMisleadingUILaunchTime)]
        ]).sample(frac=sample_frac, random_state=0)
    try:
        data_ratings1 = pd.read_parquet(
            os.path.join(C.data_path, "ratings-00001.parquet"), 
            engine="pyarrow", 
            filters=[
                [('createdAtMillis', '>', deletedNoteTombstonesLaunchTime), ('createdAtMillis', '>', notMisleadingUILaunchTime)]
            ]).sample(frac=sample_frac, random_state=0)
        data_ratings = pd.concat([data_ratings, data_ratings1]).reset_index(drop=True)

        del data_ratings1
    except:
        pass

    data_ratings['helpfulnessLevel'] = np.where(
        data_ratings['helpfulnessLevel'] == "HELPFUL",
        1,
        np.where(data_ratings['helpfulnessLevel'] == "NOT_HELPFUL", 0, 0.5)
        )
    data_ratings.rename(columns={'helpfulnessLevel': 'helpfulNum'}, inplace=True)

    return data_ratings.drop_duplicates()

def sample_user_data(
        sample_n,
        drop_ids=None,
        data_path=C.data_path, 
        random_seed=0):
    
    filters = [('enrollmentState', '==', "newUser"), ('modelingPopulation', '==', "CORE"), ('modelingGroup', '==', 0)]

    if drop_ids:
        filters.append(('participantId', 'not in', drop_ids))

    data_user_enrollment = pd.read_parquet(
        os.path.join(data_path, "userEnrollment-00000.parquet"),
        filters=[filters]).drop_duplicates()

    user_data_features = data_user_enrollment.sample(n=sample_n, random_state=random_seed)
    pid = user_data_features["participantId"].to_list()

    try:
        u = pd.read_pickle(C.users_path)
        user_data_features = pd.concat([u, user_data_features])
        user_data_features.to_pickle(C.users_path)
    
    except:
        user_data_features.to_pickle(C.users_path)

    return pid


def sample_note_data(sample_n, data_path=C.data_path, random_seed = 0):
    data_notes = pd.read_parquet(
        os.path.join(data_path, "notes-00000.parquet")).drop_duplicates()
    note_features = data_notes.sample(n=sample_n, random_state=random_seed).reset_index(drop=True)

    note_features.to_pickle(C.notes_path)
    note_status_data_to_pickle(note_features)
    return note_features["noteId"].to_list(), note_features["createdAtMillis"].to_list()

def note_status_data_to_pickle(note_features):
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
    note_status.to_pickle(C.status_path)

def generate_ratings_data(
        simulated_ratings, 
        SimModel,
        return_data=False,
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
    sim_ratings_df["helpfulNum"] = _["rating"]
    sim_ratings_df["createdAtMillis"] = _["createdAtMillis"]

    if verbose:
        print("Sampling explanation tags...")
    helpful_tags = [c for c in ratings_data.columns if c.startswith("helpful") and c != "helpfulNum"]
    nothelpful_tags = [c for c in ratings_data.columns if c.startswith("notHelpful")]

    # Indices
    data_helpful_idx = ratings_data["helpfulNum"] == 1
    data_nothelpful_idx = ratings_data["helpfulNum"] == 0

    sim_helpful_idx = sim_ratings_df["helpfulNum"] == 1
    sim_nothelpful_idx = sim_ratings_df["helpfulNum"] == 0

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

    sim_ratings_df = sim_ratings_df.fillna(sim_tag_df)
    if return_data:
        return sim_ratings_df
    
    sim_ratings_df.to_pickle(C.ratings_path)


def run_main(
        ratings_path = None,
        return_data = False,
        simulation_model = None,
        runParallel = True,
        shouldFilterNotMisleadingNotes = False,
        pseudoraters = False, 
        enabledScorers = None, 
        strictColumns = True, 
        logging = False
        ):

    scoring_dir = __import__(C.sourcecode_import_path+".scoring", fromlist=["process_pickle", "run_scoring"])

    process_pickle = scoring_dir.process_pickle
    LocalDataLoader = process_pickle.LocalDataLoader
    write_tsv_local = process_pickle.write_tsv_local

    run_scoring = scoring_dir.run_scoring
    run_scoring = run_scoring.run_scoring

    """
    Just what their main() function does...
    """
    # Load input dataframes.
    if ratings_path is None:
        dataLoader = LocalDataLoader(
            C.notes_path, 
            C.ratings_path, 
            C.status_path, 
            C.users_path, 
            headers=True,
            shouldFilterNotMisleadingNotes=shouldFilterNotMisleadingNotes)
        _, ratings, statusHistory, userEnrollment = dataLoader.get_data()
    else:
        dataLoader = LocalDataLoader(
            C.notes_path, 
            ratings_path, 
            C.status_path, 
            C.users_path, 
            headers=True,
            shouldFilterNotMisleadingNotes=shouldFilterNotMisleadingNotes,
            logging=logging)
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

    if return_data:
        exp_name = ratings_path.split("/")[-1].split(".")[0]
        helpfulnessScores = helpfulnessScores.groupby("raterParticipantId").tail(1)
        helpfulnessScores = helpfulnessScores[~helpfulnessScores["coreRaterIntercept"].isna()]
        n, u = factorization_results(simulation_model, (scoredNotes, helpfulnessScores))
        return exp_name, n, u
    
    # Write outputs to local disk.
    write_tsv_local(scoredNotes, C.scored_notes_path)
    write_tsv_local(helpfulnessScores, C.helpfulness_scores_path)
    write_tsv_local(newStatus, C.note_status_history_path)
    write_tsv_local(auxNoteInfo, C.aux_note_info_path)
