import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import content_moderation_constants as C

def results_data_load():
    note_params = pd.read_csv(C.scored_notes_path, sep="\t")
    rater_params = pd.read_csv(C.helpfulness_scores_path, sep="\t").groupby("raterParticipantId").tail(1)
    rater_params = rater_params[~rater_params["coreRaterIntercept"].isna()]
    return note_params, rater_params

def param_text(configs, note_params=None, rater_params=None):
    if note_params is not None:
        params_text = '\n'.join([
            fr'# users={configs["user_n"]}, # notes={configs["note_n"]}, R fraction={configs["user_pi_R"]:0.03f}, $\mu$={configs["note_mu"]}, cov={configs["note_cov"]}',
            fr'$o_R$={configs["user_opinion_R"]}, $o_L$={configs["user_opinion_L"]}, o_R_std={configs["user_opinion_R_std"]}, o_L_std={configs["user_opinion_L_std"]},',
            f'# R rater/note={configs["rating_R_n"]}, # L rater/note={configs["rating_L_n"]}',
            f"note_params.shape={note_params.shape}, rater_params.shape={rater_params.shape}",
        ])
    else:
        params_text = '\n'.join([
            fr'# users={configs["user_n"]}, # notes={configs["note_n"]}, R fraction={configs["user_pi_R"]:0.03f}, $\mu$={configs["note_mu"]}, cov={configs["note_cov"]}',
            fr'$o_R$={configs["user_opinion_R"]}, $o_L$={configs["user_opinion_L"]}, o_R_std={configs["user_opinion_R_std"]}, o_L_std={configs["user_opinion_L_std"]},',
            f'# R rater/note={configs["rating_R_n"]}, # L rater/note={configs["rating_L_n"]}',
        ])
    return params_text

def factorization_results(
        SimModel,
        results_data: tuple=None,
        verbose = True):
    
    if results_data is None:
        note_params, rater_params = results_data_load()
    else:
        note_params, rater_params = results_data

    if verbose:
        print("Loading factorization results datasets")
        print("\t Note parameters shape: ", note_params.shape)
        print("\t Rater parameters shape: ", rater_params.shape)

    # User
    sim_users = pd.DataFrame(SimModel.user_set_I, columns=["user_sim_ID", "dim_accuracy", "dim_value"])\
        .merge(pd.DataFrame(SimModel.user_id_map), on="user_sim_ID", how="left")
    user_input_output = sim_users.merge(rater_params, left_on="participantId", right_on="raterParticipantId")
    user_input_output["Group"] = np.where(user_input_output["dim_value"] > 0, "R", "L")
    user_input_output.rename({"coreRaterFactor1": "est_dim_value", "coreRaterIntercept": "est_dim_accuracy"}, axis=1, inplace=True)

    flipped = user_input_output.loc[user_input_output["Group"]=="R", "est_dim_value"].min() < 0 # Flipped group factor

    # Note
    sim_notes = pd.DataFrame(SimModel.note_set_J, columns=["note_sim_ID", "dim_accuracy", "dim_value"])\
        .merge(pd.DataFrame(SimModel.note_id_map), on="note_sim_ID")\
            .drop("createdAtMillis", axis=1)
    note_input_output = sim_notes.merge(note_params, on="noteId")

    note_input_output.rename({"coreNoteFactor1": "est_dim_value", "coreNoteIntercept": "est_dim_accuracy"}, axis=1, inplace=True)
    note_input_output["output"] = np.where(note_input_output["finalRatingStatus"] == "CURRENTLY_RATED_HELPFUL", 1,
                                           np.where(note_input_output["finalRatingStatus"] == "CURRENTLY_RATED_NOT_HELPFUL", 0, 0.5))

    if flipped:
        note_input_output["est_dim_value"] = -note_input_output["est_dim_value"]
        user_input_output["est_dim_value"] = -user_input_output["est_dim_value"]

    return note_input_output, user_input_output

def plot_factorization_results(
        note_input_output, 
        user_input_output,
        configs=None,
        note_sample_n=None, 
        note_sample_frac=1, 
        user_sample_n=None, 
        user_sample_frac=1):
    """
    Args:
        note_input_output: matrix factorization results with simulation data (pd.DataFrame)
        user_input_output: matrix factorization results with simulation data (pd.DataFrame)
        note_sample_n: number of notes to sample (int)
        note_sample_frac: fraction of notes to sample (float)
        user_sample_n: number of users to sample (int)
        user_sample_frac: fraction of users to sample (float)
    """
    plt.rcParams['font.family'] = 'monospace'
    params_text = param_text(configs, note_input_output, user_input_output)

    note_input_output_to_plot = note_input_output.sample(n=note_sample_n, frac=note_sample_frac)
    user_input_output_to_plot = user_input_output.sample(n=user_sample_n, frac=user_sample_frac)
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    sub_L = sns.scatterplot(data=note_input_output_to_plot, x="est_dim_value", y="est_dim_accuracy", hue="H_L", ax=axs[1,0], legend=False)
    sub_All = sns.scatterplot(data=note_input_output_to_plot, x="est_dim_value", y="est_dim_accuracy", hue="H_all", ax=axs[0,0], legend=False)
    sub_R = sns.scatterplot(data=note_input_output_to_plot, x="est_dim_value", y="est_dim_accuracy", hue="H_R", ax=axs[0,1], legend="auto")

    sns.move_legend(sub_R, "upper left", bbox_to_anchor=(1, 1), title=None)
    fig.text(x=0.5, y=0.92, s=params_text, fontsize=9, ha='center', va='bottom')

    title_font = {'style': 'italic',
                        'weight': 'heavy', 
                        'size': 9}
    sub_L.set_title('hue = support in group L', fontdict=title_font)
    sub_All.set_title('hue = support in the population', fontdict=title_font)
    sub_R.set_title('hue = support in group R', fontdict=title_font)

    rater_scatter = sns.scatterplot(data=user_input_output_to_plot, x="est_dim_value", y="est_dim_accuracy", hue="Group", ax=axs[1,1])
    rater_scatter.set_title('Rater results', fontdict=title_font)

    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    sns.scatterplot(data=note_input_output_to_plot, x="dim_value", y="dim_accuracy", hue="H_all", ax=axs[0])
    sns.scatterplot(data=note_input_output_to_plot, x="dim_value", y="dim_accuracy", hue="output", ax=axs[1])

    fig.text(x=0.5, y=0.92, s=params_text, fontsize=9, ha='center', va='bottom')

    plt.show()
    

def plot_distance_results(x, y, configs=None, text_y=1.3):
    from scipy import stats

    plt.rcParams['font.family'] = 'monospace'

    coefficients = np.polyfit(x, y, deg=4)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)

    y_err = x_fit.std() * np.sqrt(1/len(x_fit) + (x_fit - x_fit.mean())**2 / np.sum((x_fit - x_fit.mean())**2))
    ci = stats.norm.interval(0.95, loc=y_fit, scale=y_err)  # 95% confidence interval

    sns.scatterplot(x=x, y=y, color="lightpink")
    sns.lineplot(x=x_fit, y=y_fit, color='crimson')
    if configs:
        params_text = param_text(configs)
        plt.text(x=0, y=max(y)*text_y, s=params_text, fontsize=9, ha='center', va='bottom')
    plt.fill_between(x_fit, ci[0], ci[1], color='pink', alpha=.1)

    plt.show()

def plot_user_location(strategic_id, user_input_output, experiment_input_output):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
    cmap = plt.cm.get_cmap('cool')

    non_strategic = user_input_output.loc[user_input_output["user_sim_ID"] != strategic_id,:]
    num_experiment = experiment_input_output.shape[0]

    value = sns.scatterplot(data=non_strategic, x="dim_value", y="est_dim_value", c='pink', label='other players', ax=axs[0])
    accuracy = sns.scatterplot(data=non_strategic, x="dim_accuracy", y="est_dim_accuracy", c='pink', label='other players', ax=axs[1], legend=False)

    for i in range(num_experiment):
        color = cmap(i / (num_experiment))
        this_exp = experiment_input_output.iloc[i,:]
        strategy = this_exp["strategy"]
        
        sns.scatterplot(x=[this_exp["dim_value"]], y=[this_exp["est_dim_value"]], c=[color], label=strategy, ax=axs[0], zorder=10)
        sns.scatterplot(x=[this_exp["dim_value"]], y=[this_exp["est_dim_value"]], c=[color], label=strategy, ax=axs[1], zorder=10)

    x = np.linspace(user_input_output['est_dim_value'].min(), user_input_output['est_dim_value'].max(), 10)
    sns.lineplot(x=x, y=x, color="#ff42a4", alpha=0.5, label='45 degree line', ax=axs[0])

    value.set(xlabel='True value dimension', ylabel='Estimated value dimension')
    accuracy.set(xlabel='True accuracy dimension', ylabel='Estimated accuracy dimension')
    accuracy.yaxis.tick_right()
    accuracy.yaxis.set_label_position("right")
    fig.tight_layout()

    plt.show()

def experiment_concat(U, rating_input, target, user_input_output, note_input_output, strategy_label="", experiment_input_output=pd.DataFrame(), reset_data=False):
    strategy = pd.DataFrame(rating_input.iloc[target,:]).T.reset_index(drop=True)[['note_sim_ID', 'note_dim_accuracy', 'note_dim_value', 'user_sim_ID', 'evaluation', 'rating']]
    print(strategy['note_sim_ID'][0])
    strategy['note_est_dim_accuracy'], strategy['note_est_dim_value'], strategy['note_output'] = note_input_output.loc[note_input_output['note_sim_ID'] == strategy['note_sim_ID'][0], ['est_dim_accuracy', 'est_dim_value', 'output']].to_numpy()[0]
    strategy['strategy'] = strategy_label
    new_experiment_input_output = strategy.merge(user_input_output[user_input_output['user_sim_ID'] == U.sim_id], on="user_sim_ID")
    if not reset_data:
        new_experiment_input_output = pd.concat([experiment_input_output, new_experiment_input_output])
    return new_experiment_input_output

def smoothed_dot(note_df, user_o, s, t):
    probability = 1/(1+np.exp(-s*(note_df["est_dim_accuracy"].to_numpy() - t)))
    return ((note_df[['dim_accuracy', 'dim_value']] * user_o).sum(axis=1) * probability).mean()

def smoothed_prob(note_accuracy_dim, s, t):
    return 1/(1+np.exp(-s*(note_accuracy_dim - t)))

def get_exp_diff(user_output: dict, note_output: dict, s_list = None, threshold_list = None, U = None):
    user_delta_df = pd.DataFrame()
    target_delta_df = pd.DataFrame()

    baseline_user_df = user_output['baseline'].dropna(axis=1, how="all").select_dtypes(include='number')
    baseline_note_df = note_output['baseline'].dropna(axis=1, how="all").select_dtypes(include='number')
    for s in s_list:
        for t in threshold_list:
            baseline_note_df[f"output_prob_s={s}_t={t}"] = smoothed_dot(baseline_note_df, U.opinion, s, t)
    baseline_dot = {f"s={s}_t={t}": smoothed_dot(baseline_note_df, U.opinion, s, t) for s in s_list for t in threshold_list}

    for exp_name in user_output.keys():
        if exp_name == "baseline":
            continue
        
        # User output
        user_df = user_output[exp_name].dropna(axis=1, how="all").select_dtypes(include='number')
        note_df = note_output[exp_name].dropna(axis=1, how="all").select_dtypes(include='number')
            
        _, target_note_id, eval, rating = exp_name.split("_")
        target_note_id = int(target_note_id)

        user_delta_df.loc[f"b_{target_note_id}", "experiment"] = False
        user_delta_df.loc[f"b_{target_note_id}", "target_note_id"] = target_note_id
        user_delta_df.loc[f"b_{target_note_id}", "evaluation"] = eval
        user_delta_df.loc[f"b_{target_note_id}", "rating"] = eval

        user_delta_df.loc[exp_name, "experiment"] = True
        user_delta_df.loc[exp_name, "target_note_id"] = target_note_id
        user_delta_df.loc[exp_name, "evaluation"] = eval
        user_delta_df.loc[exp_name, "rating"] = rating

        for s in s_list:
            for t in threshold_list:
                user_delta_df.loc[f"b_{target_note_id}", f"output_dot_s={s}_t={t}"] = baseline_dot[f's={s}_t={t}'] - baseline_dot[f's={s}_t={t}']
                user_delta_df.loc[exp_name, f"output_dot_s={s}_t={t}"] = smoothed_dot(note_df, U.opinion, s, t) - baseline_dot[f's={s}_t={t}']
    
        experiment_output = user_df.loc[user_df['user_sim_ID']==U.sim_id,:].to_dict('records')[0]
        counterfactual_output = baseline_user_df.loc[baseline_user_df['user_sim_ID']==U.sim_id,:].to_dict('records')[0]
        
        for c in list(experiment_output.keys())[1:]:
            user_delta_df.loc[f"b_{target_note_id}", c] = counterfactual_output[c] - counterfactual_output[c]
            user_delta_df.loc[exp_name, c] = experiment_output[c] - counterfactual_output[c]
    
        # Note output
        target_delta_df.loc[f"b_{target_note_id}", "experiment"] = False
        target_delta_df.loc[f"b_{target_note_id}", "target_note_id"] = target_note_id
        target_delta_df.loc[f"b_{target_note_id}", "evaluation"] = eval
        target_delta_df.loc[f"b_{target_note_id}", "rating"] = eval

        target_delta_df.loc[exp_name, "experiment"] = True
        target_delta_df.loc[exp_name, "target_note_id"] = target_note_id
        target_delta_df.loc[exp_name, "evaluation"] = eval
        target_delta_df.loc[exp_name, "rating"] = rating

        experiment_output = note_df.loc[note_df['note_sim_ID'] == target_note_id,:].to_dict('records')[0]
        counterfactual_output = baseline_note_df.loc[baseline_note_df['note_sim_ID'] == target_note_id,:].to_dict('records')[0]

        for s in s_list:
            for t in threshold_list:
                target_delta_df.loc[f"b_{target_note_id}", f"output_s={s}_t={t}"] = smoothed_prob(counterfactual_output['est_dim_accuracy'], s, t) - smoothed_prob(counterfactual_output['est_dim_accuracy'], s, t) 
                target_delta_df.loc[exp_name, f"output_s={s}_t={t}"] = smoothed_prob(experiment_output['est_dim_accuracy'], s, t) - smoothed_prob(counterfactual_output['est_dim_accuracy'], s, t) 

        for c in list(experiment_output.keys())[1:]:
            target_delta_df.loc[f"b_{target_note_id}", c] = counterfactual_output[c] - counterfactual_output[c]
            target_delta_df.loc[exp_name, c] = experiment_output[c] - counterfactual_output[c]
    user_delta_df = user_delta_df.dropna(axis=1, how="all")
    target_delta_df = target_delta_df.dropna(axis=1, how="all")
    user_delta_df = user_delta_df[[c for c in user_delta_df.columns if not c.startswith("timestamp")]]
    target_delta_df = target_delta_df.drop(["createdAtMillis", 'noteId', 'numRatings'], axis=1)
    return user_delta_df, target_delta_df