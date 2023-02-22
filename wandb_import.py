import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FROM_CSV = True
PLOT_REWARD = True # True: reward False: grad variance
PLOT_3by3 = False
PLOT_TIME = False
CSV_PATH = "pgtree_results.h5"
store = pd.HDFStore(CSV_PATH)

if not FROM_CSV:
    api = wandb.Api()
    project_name = "pg-tree"
    runs = api.runs("nvr-israel/{}".format(project_name))
    summary_list = []
    config_list = []
    name_list = []
    steps_list = []
    timestamps_list = []
    rewards_list = []
    variances_list = []
    relevant_descriptions = ["Baseline of multiple environments PPO fixed episodic", "All games all depths"]
    for run in runs:
        # if run.sweep is not None and run.sweep.id in sweep_ids:
        if "experiment_description" in run.config and run.config["experiment_description"] in relevant_descriptions:
            h_df = run.history(keys=["train\episodic_reward", "_timestamp", "train\policy_weights_grad_var"], samples=3000) # can either use run.history() or, instead, scan_history() and build list of dicts via [r for r in h_df]
            hist_dicts = [r for r in h_df]
            if len(hist_dicts) > 0: # len(h_df) > 0:
                # run.summary are the output key/values like accuracy.
                # We call ._json_dict to omit large files
                summary_list.append(run.summary._json_dict)

                # run.config is the input metrics.
                # We remove special values that start with _.
                config = {k:v for k,v in run.config.items() if not k.startswith("_")}
                config["sweep_id"] = run.sweep.id
                config_list.append(config)

                # run.name is the name of the run.
                name_list.append(run.name)

                # history_list.append(h_df.filter(["_step", "reward"]).to_records(index=False)) #df = df.astype("object")

                # steps_list.append(h_df.filter(["_step"]))
                # rewards_list.append(list(h_df.filter(["reward"]).reward))
                env_name = config_list[0]["env_name"]
                rew_str = "reward"
                # steps = [e["_step"] for e in hist_dicts]
                # rewards = [e[rew_str] for e in hist_dicts]
                steps = h_df._step.to_list()
                timestamps = h_df._timestamp.to_list()
                rewards = h_df["train\episodic_reward"].to_list()
                variances = h_df["train\policy_weights_grad_var"].to_list()
                steps_list.append(steps)
                timestamps_list.append(timestamps)
                rewards_list.append(rewards)
                variances_list.append(variances)
                # rewards_list.append(list(h_df.filter(["reward"]).reward))

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    # history_df = pd.DataFrame.from_records(history_list)
    reward_df = pd.DataFrame({"reward_vec": [np.nan] * len(name_df)})
    reward_df["reward_vec"] = reward_df["reward_vec"].astype("object")
    for i_r, r in enumerate(rewards_list):
        reward_df.reward_vec[i_r] = r
    step_df = pd.DataFrame({"step_vec": [np.nan] * len(name_df)})
    step_df["step_vec"] = step_df["step_vec"].astype("object")
    for i_step, s in enumerate(steps_list):
        step_df.step_vec[i_step] = s
    timestamp_df = pd.DataFrame({"timestamp_vec": [np.nan] * len(name_df)})
    timestamp_df["timestamp_vec"] = timestamp_df["timestamp_vec"].astype("object")
    for i_timestamp, ts in enumerate(timestamps_list):
        timestamp_df.timestamp_vec[i_timestamp] = ts
    variance_df = pd.DataFrame({"variance_vec": [np.nan] * len(name_df)})
    variance_df["variance_vec"] = variance_df["variance_vec"].astype("object")
    for i_variance, v in enumerate(variances_list):
        variance_df.variance_vec[i_variance] = v

    all_df = pd.concat([name_df, config_df, summary_df, reward_df, step_df, timestamp_df, variance_df], axis=1)
    store["all_df"] = all_df
else:
    # all_df = pd.read_csv(CSV_PATH)
    all_df = store["all_df"]
    # store["df"] = all_df

convergence_plots = True
if convergence_plots and FROM_CSV:
    def moving_average(x, alpha):
        out = [x[0]]
        c = 1
        for i in range(len(x) - 1):
            new_val = alpha * out[-1] + (1 - alpha) * x[c]
            out.append(new_val)
            c += 1
        return np.asarray(out)
    def moving_average2(x, w=20):
        return np.convolve(x, np.ones(w), "same") / np.convolve(x * 0 + 1, np.ones(w), "same") # w

    depths_to_ignore = [1, 4, 7]

    import matplotlib
    import matplotlib.colors as mcol
    import matplotlib.cm as cm

    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])

    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(vmin=1, vmax=8)

    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])

    line_width = 1.5
    fontsize = 18
    w_size = 30000 if PLOT_TIME else 30
    matplotlib.rcParams.update({"font.size": fontsize - 4})
    game_envs = ["Asteroids", "Breakout", "VideoPinball", "KungFuMaster", "Phoenix",
                 "Gopher", "Krull", "NameThisGame"]#, "CrazyClimber", "RoadRunner"]
    game_envs = ["Asteroids"]
    if PLOT_3by3:
        game_envs.append("CrazyClimber")
    game_envs_full = [n + "NoFrameskip-v4" for n in game_envs]
    ylim_dict = {"Breakout": 350, "Asteroids": 5000, "VideoPinball": 400000, "SpaceInvaders": 1200, "MsPacman": 2500}
    # figure, big_axes = plt.subplots(nrows=1, ncols=len(game_envs))
    if PLOT_3by3:
        nrows = 3; ncols = 3
    else:
        nrows = 2; ncols = 4
    figure, big_axes = plt.subplots(figsize=(11.0, 7.0), nrows=nrows, ncols=ncols) #  sharey=True)

    # for i in range(len(game_envs)):
    #     big_axes[i].tick_params(labelcolor=(1., 1., 1., 0.0), top="off", bottom="off", left="off", right="off")
    #     # removes the white frame
    #     big_axes[i]._frameon = False

    for i in range(len(big_axes)):
        for j in range(len(big_axes[0])):
            big_axes[i][j].set_xticks([])
            big_axes[i][j].set_yticks([])
    # plt.tick_params(
    #     axis="x",  # changes apply to the x-axis
    #     which="both",  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off

    alpha_smoothing = 0.9

    plot_count = 0
    df_envs = all_df.groupby(["env_name"])
    for i_env, env_name in enumerate(df_envs.groups):
        if env_name not in game_envs_full:
            continue
        plot_count += 1
        if PLOT_3by3:
            ax = figure.add_subplot(3, 3, plot_count)
        else:
            ax = figure.add_subplot(2, 4, plot_count)
        df_env = df_envs.get_group(env_name)
        df_depths = df_env.groupby("tree_depth")
        for i_depth, depth in enumerate(df_depths.groups):
            if depth in depths_to_ignore:
                continue
            df_depth = df_depths.get_group(depth)
            if PLOT_REWARD:
                min_y_len = min(df_depth.reward_vec.str.len()) - w_size
            else:
                min_y_len = min(df_depth.variance_vec.str.len()) - w_size
            # smallest_x = min([l[-1] for l in df_depth.step_vec])
            if PLOT_TIME:
                largest_x = max([l[-1] - l[0] for l in df_depth.timestamp_vec])
            else:
                largest_x = df_depth["_step"].max()
            x_vals_shared = np.linspace(1, largest_x, round(largest_x / 3))

            y_vals_vec = None
            for i_run in range(df_depth.shape[0]):  # iterate on seeds
                print("Seed: {}".format(i_run))
                if PLOT_REWARD:
                    y_vals = df_depth.iloc[i_run].reward_vec
                else:
                    y_vals = df_depth.iloc[i_run].variance_vec
                if PLOT_TIME:
                    x_vals = np.asarray(df_depth.iloc[i_run].timestamp_vec) - np.asarray(df_depth.iloc[i_run].timestamp_vec)[0]
                else:
                    x_vals = np.asarray(df_depth.iloc[i_run].step_vec) - np.asarray(df_depth.iloc[i_run].step_vec)[0] + 1
                    y_vals = np.asarray(y_vals)
                # f = interp1d(x_vals, y_vals, fill_value="extrapolate")
                f = interp1d(x_vals, y_vals)
                last_loc = np.where(x_vals_shared >= x_vals[-1])[0][0]
                x_vals = x_vals_shared[:last_loc]
                y_vals = f(x_vals)
                y_vals = moving_average2(y_vals, w=w_size)
                w_drop = 10000 #if PLOT_TIME else 10
                nans = np.empty(x_vals_shared.shape[0] - len(y_vals))
                nans[:] = np.nan
                y_vals = np.concatenate((y_vals, nans))

                if y_vals_vec is None:
                    y_vals_vec = y_vals
                else:
                    y_vals_vec = np.vstack((y_vals_vec, y_vals))
            label = "Depth {}".format(depth)
            if depth == 0:
                color = "g"
                final_label = "PPO"
                lw = line_width + 1
            else:
                color = cpick.to_rgba(depth)
                final_label = "SoftTreeMax " + label
                lw = line_width
            x_vec = x_vals_shared[:-w_drop] / 3600 if PLOT_TIME else x_vals_shared[:-w_drop]
            if len(y_vals_vec.shape) > 1:
                y_mean = np.nanmean(y_vals_vec, axis=0)
                y_std = np.nanstd(y_vals_vec, axis=0)
                under_line = (y_mean - y_std)
                over_line = (y_mean + y_std)
                ax.plot(x_vec, y_mean[:-w_drop], linewidth=lw, label=final_label, color=color)
                ax.fill_between(x_vec, under_line[:-w_drop], over_line[:-w_drop], color=color, alpha=.15)
                if PLOT_TIME:
                    ax.set_xlim([0, 168])
                    ax.set_xlabel("Time [hours]", fontsize=fontsize)
                else:
                    ax.set_xlabel("Environment Steps (log scale)", fontsize=fontsize)
                # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.grid("on")
                ax.set_title(env_name[:-14], fontsize=fontsize)
            else:
                ax.plot(x_vec, y_vals_vec[:-w_drop], linewidth=lw, label=final_label, color=color)
            if not PLOT_REWARD:
                ax.set_yscale("log")
            if not PLOT_TIME:
                ax.set_xscale("log")
            yskip = [1, 4, 7] if PLOT_3by3 else [1, 5]
            if plot_count in yskip:
                if PLOT_REWARD:
                    ax.set_ylabel("Reward", fontsize=fontsize)
                else:
                    ax.set_ylabel("Gradient variance\n(log scale)", fontsize=fontsize)
            print("finished depth {}".format(depth))
            # if plot_count == 1:
            #     plt.legend(framealpha=1)

    print("finished")
    plt.show()

    figure.set_facecolor("w")
    plt.tight_layout()