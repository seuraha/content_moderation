import numpy as np
from numpy.random import permutation, exponential, multivariate_normal, binomial
import pandas as pd
import time

def simulate_ratings(n_rater_R: int, n_rater_L: int, set_notes: pd.DataFrame, set_I_R: pd.DataFrame, set_I_L: pd.DataFrame):
    """
    Takes note opinion feature data, group opinion data,
     assigns raters from each group to each note,
     and returns a dataframe of note opinion, assigned raters opinion, and ratings

    Args:
        n_rater_R: number of raters from group R per note (int)
        n_rater_L: number of raters from group L per note (int)
        set_notes: set of notes, matrix of dimension `n_notes` X 3 (pd.DataFrame)
        set_I_R: set of users in group R, matrix of dimension `n_R` X 3 (pd.DataFrame)
        set_I_L: set of users in group L, matrix of dimension `n_L` X 3 (pd.DataFrame)

    Returns: 
        a dataset containing notes, assigned raters, and the ratings (pd.DataFrame)
    """

    # Parameters
    n = n_rater_R + n_rater_L   # total number of raters per note
    rater_column_names = [f"rater{i}" for i in range(n)]

    # Random assignment of `n` raters to each note
    assigned_raters = pd.DataFrame(
        set_notes["note_sim_ID"]\
            .apply(lambda x: np.vstack((permutation(set_I_R)[:n_rater_R], permutation(set_I_L)[:n_rater_L])).tolist())\
                .tolist(), 
        columns=rater_column_names
        )

    # Reshaping table (wide to long)
    J_assigned_rater = pd.concat([set_notes, assigned_raters], axis=1)
    J_assigned_rater = pd.melt(
        J_assigned_rater,
        id_vars=['note_sim_ID', 'dim_T', 'dim_R'],
        value_vars=rater_column_names,
        value_name='rater_data'
    )
    J_assigned_rater[['rater_sim_ID', 'rater_dim_T', 'rater_dim_R']] = pd.DataFrame(J_assigned_rater['rater_data'].tolist(), index=J_assigned_rater.index)
    J_assigned_rater = J_assigned_rater.drop(['variable', 'rater_data'], axis=1)

    # Rating 
    # Decision rule: 
    #   helpful (1)       if dot product >= 0.5,
    #   somewhat (0.5)    if 0 <= dot product < 0.5,
    #   not helpful (0)   if dot product < 0
    dot = J_assigned_rater["dim_T"]*J_assigned_rater["rater_dim_T"]+J_assigned_rater["dim_R"]*J_assigned_rater["rater_dim_R"]
    J_assigned_rater["dot"] = dot
    J_assigned_rater["rating"] = np.where(dot>=0.5, "HELPFUL", np.where(dot>=0, "SOMEWHAT_HELPFUL", "NOT_HELPFUL"))
    J_assigned_rater["rating"] = J_assigned_rater["rating"].astype("string")

    return J_assigned_rater

def simulate_explanation_tags(ratings_data: pd.DataFrame, sim_ratings_data: pd.DataFrame, helpful_tags: list, nothelpful_tags: list):
    """
    In addition to helpfulness, Community Notes maintains tags to explain why users chose such a rating.
        e.g.,   if rated helpful, whether it was because it was clear, informative, or provides important contexts
                if rated not helpful, whether it was because it was outdated, unclear, or off topic.
                
    Wherever the rating is "helpful," simulates tags indicating reasons for rating helpful (binary);
    and wherever the rating is "not helpful," simulates tags indicating reasons for rating not helpful (binary);
    0, elsewhere.

    Args:
        ratings_data: ratings data from Community Notes (pd.DataFrame)
        sim_ratings_data: simulated rating data (pd.DataFrame)
        helpful_tags: column names of explanation tags for "helpful" ratings
        nothelpful_tags: column names of explanation tags for "not helpful" ratings
        
    Returns: 
        a dataset with columns `helpful_tags`+`nothelpful_tags` containing simulated tags (pd.DataFrame)
    """

    # Indices
    data_helpful_idx = ratings_data["helpfulnessLevel"] == "HELPFUL"
    data_nothelpful_idx = ratings_data["helpfulnessLevel"] == "NOT_HELPFUL"

    sim_helpful_idx = sim_ratings_data["helpfulnessLevel"] == "HELPFUL"
    sim_nothelpful_idx = sim_ratings_data["helpfulnessLevel"] == "NOT_HELPFUL"

    # Create DataFrame with zeros
    sim_zeros = np.zeros((sim_ratings_data.shape[0],len(helpful_tags + nothelpful_tags)))
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

    return sim_tag_df


def simulation_data_generation(
        configs: dict, 
        original_datasets: dict
        ):
    """
    Returns 4 dataset inputs for the algorithm

    Args:
        configs: parameters (dict)
        original_datasets: Community Notes datasets (dict of pd.DataFrame)

    Returns: 
        datasets: tuple of (sim_ratings, users, notes, sim_ratings_data, sim_note_status)
            sim_ratings: simulated ratings (notes X users) (pd.DataFrame)
            users: randomly sampled features of users attached to simulated users (pd.DataFrame)
            notes: randomly sampled features of notes attached to simulated notes (pd.DataFrame)
            sim_ratings_data: ratings data with explanation tags
            sim_note_status: status history data of simulated notes
    """
    n_users, n_notes = configs["n_users"], configs["n_notes"]
    pi_R = configs["pi_R"]
    mu, cov = configs["notes_mu"], configs["notes_cov"]
    o_r, o_l = configs["opinion_R"], configs["opinion_L"]
    o_R_std, o_L_std = configs["opinion_R_std"], configs["opinion_L_std"]
    n_rater_R, n_rater_L = configs["n_rater_R"], configs["n_rater_L"]

    # I, I_R, I_L = get_user_set(n_users, pi_R, o_r, o_l)
    I, I_R, I_L = get_user_set(n_users, pi_R, o_r, o_l, o_R_std, o_L_std)
    J = get_note_set(n_notes, mu, cov)

    data_user_enrollment, data_notes, data_ratings, data_note_history = original_datasets["users"], original_datasets["notes"], original_datasets["ratings"], original_datasets["note_history"]
    sim_ratings = simulate_ratings(n_rater_R, n_rater_L, J, I_R, I_L)
    avg_wait_time = 2*60*60*1000 # a note gets a rating once every 2 hours - this is not essential but to add "created at millisec" column to the rating data.

    # 1. User enrollment data
    #   randomly samples `n_users` data points from the original Community Notes dataset 
    #   and concatenates them to the simulated user dataset.
    sampled_user_data = data_user_enrollment.sample(n=n_users, random_state=0).reset_index(drop=True)
    users = pd.concat([I, sampled_user_data], axis=1)

    # 2. Note data
    #   randomly samples `n_notes` data points from the original Community Notes dataset 
    #   and concatenates them to the simulated note dataset.
    sampled_note_data = data_notes.sample(n=n_notes, random_state=0).reset_index(drop=True)
    notes = pd.concat([J, sampled_note_data], axis=1)

    # 3. Ratings data
    #   columns `noteId`, `raterParticipantId`, `helpfulnessLevel`: pasted from the simulated data
    #   `createdAtMillis`: original note created time + simulated wait time interval as a Poisson process
    #   `version`: 2, version 1 is deprecated in 2021. The main difference is the existence of "somewhat helpful" rating.
    #   deprecated v1 features `agree`, `disagree`, `helpful`, `notHelpful`: seems to be not used in version 2. Set to the default values
    #   `ratedOnTweetId`: some data include their Tweet ID but most data are set as -1. I pasted -1 to all rows.
    #   explanation tags: each column simulated following the statistics of the original dataset
    sim_ratings_data = pd.DataFrame(columns=data_ratings.columns)
    _ = sim_ratings.merge(users[["user_sim_ID", "participantId"]], left_on="rater_sim_ID", right_on="user_sim_ID", how="left")\
        .merge(notes[["note_sim_ID", "noteId", "createdAtMillis"]], on="note_sim_ID", how="left")
    _["createdAtMillis"] = _["createdAtMillis"].apply(lambda x: x+int(exponential(scale=avg_wait_time, size=1)[0]))
    sim_ratings_data["noteId"] = _["noteId"]
    sim_ratings_data["raterParticipantId"] = _["participantId"]
    sim_ratings_data["version"] = 2
    sim_ratings_data["agree"] = 0       # deprecated, set to 0 by default in version 2
    sim_ratings_data["disagree"] = 0    # deprecated, set to 0 by default in version 2
    sim_ratings_data["helpful"] = 0     # deprecated, set to 0 by default in version 2
    sim_ratings_data["notHelpful"] = 0  # deprecated, set to 0 by default in version 2
    sim_ratings_data["helpfulnessLevel"] = _["rating"]
    sim_ratings_data["createdAtMillis"] = _["createdAtMillis"]
    sim_ratings_data["ratedOnTweetId"] = -1

    # explanation tags
    helpful_tags = [c for c in data_ratings.columns if c.startswith("helpful") and c not in ["helpful", "helpfulnessLevel"]]
    nothelpful_tags = [c for c in data_ratings.columns if c.startswith("notHelpful") and c not in ["notHelpful"]]

    tags = simulate_explanation_tags(data_ratings, sim_ratings_data, helpful_tags, nothelpful_tags)
    sim_ratings_data = sim_ratings_data.fillna(tags)

    # 4. Note status history
    #   all notes set to be "needs more ratings"
    sim_note_status = pd.DataFrame(columns = data_note_history.columns)
    sim_note_status["noteId"] = notes["noteId"]
    sim_note_status["noteAuthorParticipantId"] = notes["noteAuthorParticipantId"]
    sim_note_status["createdAtMillis"] = notes["createdAtMillis"]
    sim_note_status["timestampMillisOfCurrentStatus"] = round(time.time()*1000) # current time
    sim_note_status["timestampMillisOfFirstNonNMRStatus"] = -1
    sim_note_status["currentStatus"] = "NEEDS_MORE_RATINGS"
    sim_note_status["timestampMillisOfLatestNonNMRStatus"] = -1
    sim_note_status["timestampMillisOfRetroLock"] = -1

    return (sim_ratings, users, notes, sim_ratings_data, sim_note_status)


def get_user_set(n_users, pi_R, o_r, o_l, o_R_std, o_L_std):
    from numpy.random import normal
    n_R = int(n_users*pi_R)     # total number of users in group R
    n_L = n_users-n_R # total number of users in group L

    r = normal(o_r[1], o_R_std, n_R).reshape((-1,1))
    i = np.array(["R"]*n_R).reshape((-1,1))
    o_r = np.hstack((1-r, r, i))

    l = normal(o_l[1], o_L_std, n_L).reshape((-1,1))
    i = np.array(["L"]*n_L).reshape((-1,1))
    o_l = np.hstack((1-abs(l), l, i))

    _ = permutation(n_users).reshape((n_users,1))               # user indices
    I_R = np.hstack((_[:n_R], o_r))   # set of users in group R: matrix of dimension `n_R` X 4 (index, accuracy dimension, value dimension, group)
    I_L = np.hstack((_[n_R:], o_l))   # set of users in group L: matrix of dimension `n_L` X 4
    I = np.vstack((I_R, I_L))                                   # set of all users: matrix of dimension `n_users` X 3
    I = pd.DataFrame(I, columns=["user_sim_ID", "dim_T", "dim_R", "Group"])
    return (I, I_R, I_L)


def get_note_set(n_notes, mu, cov):
    _ = pd.DataFrame(np.array(range(n_notes)).reshape(n_notes,1), columns = ["note_sim_ID"])    # note indices
    o = pd.DataFrame(multivariate_normal(mu, cov, n_notes), columns=["dim_T", "dim_R"])         # note opinion features: multivariate normal
    J = pd.concat([_, o], axis=1)                                                               # set of notes: matrix of dimension `n_notes` X 3
    return J
