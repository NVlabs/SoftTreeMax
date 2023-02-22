import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

PLOT_REWARD = False # True: reward False: grad variance
PLOT_3by3 = False
CSV_PATH = "pgtree_results.h5"
store = pd.HDFStore(CSV_PATH)

depths_to_ignore = [] #[1, 4, 7]

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

MAX_DEPTH = 9
line_width = 3
fontsize = 18
w_size = 30000
matplotlib.rcParams.update({"font.size": fontsize - 4})
game_envs = ["Asteroids", "Breakout", "KungFuMaster", "Phoenix",
             "Gopher", "Krull", "NameThisGame", "VideoPinball"]#, , ""CrazyClimber""]
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

for i in range(len(big_axes)):
    for j in range(len(big_axes[0])):
        big_axes[i][j].set_xticks([])
        big_axes[i][j].set_yticks([])

alpha_smoothing = 0.9

all_df = store["all_df"]
plot_count = 0
df_envs = all_df.groupby(["env_name"])
MAX_DEPTH = df_envs.tree_depth.max().max() + 1
depth_vec = np.arange(1, MAX_DEPTH)
yskip_rew = [1, 4, 7] if PLOT_3by3 else [1, 5]
yskip_var = [3, 6, 9] if PLOT_3by3 else [4, 8]
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
    reward_depth_vec = np.zeros(MAX_DEPTH)
    var_depth_vec = np.zeros(MAX_DEPTH)
    for i_depth, depth in enumerate(df_depths.groups):
        if depth in depths_to_ignore:
            continue
        df_depth = df_depths.get_group(depth)
        reward_vec = []
        var_vec = []
        for i_run in range(df_depth.shape[0]):  # iterate on seeds
            reward = df_depth.iloc[i_run].reward_vec
            var = df_depth.iloc[i_run].variance_vec
            num_samples = round(len(reward) * 0.03)
            top_ind = np.argpartition(reward, -num_samples)[-num_samples:]
            reward_vec.append(np.mean(np.asarray(reward)[top_ind]))
            var_vec.append(np.mean(var[round(len(var) * 0.2):]))
            # var_vec.append(np.mean(np.asarray(var)[top_ind]))
        reward_mean = np.mean(reward_vec)
        var_mean = np.mean(var_vec)
        reward_depth_vec[i_depth] = reward_mean
        var_depth_vec[i_depth] = var_mean
    lns1 = ax.plot(depth_vec, reward_depth_vec[1:], color="blue", marker="o", label="SoftTreeMax Reward")
    lns2 = ax.axhline(y=reward_depth_vec[0], color="blue", linestyle="dashed", label="PPO Reward")
    ax.set_xlabel("Depth", fontsize=fontsize)
    if plot_count in yskip_rew:
        ax.set_ylabel("Reward", fontsize=fontsize)
    ax2 = ax.twinx()
    lns3 = ax2.plot(depth_vec, var_depth_vec[1:], color="red", marker="o", label="SoftTreeMax Variance")
    lns4 = ax2.axhline(y=var_depth_vec[0], color="red", linestyle="dashed", label="PPO Variance")
    ax2.set_yscale("log")
    if plot_count in yskip_var:
        ax2.set_ylabel("Gradient variance\n(log scale)", fontsize=fontsize)
    # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid("on")
    ax.set_title(env_name[:-14], fontsize=fontsize)
    # if not PLOT_REWARD:
    #     ax.set_yscale("log")

    # if plot_count == 1:
    #     lns = lns1 + [lns2] + lns3 + [lns4]
    #     labs = [l.get_label() for l in lns]
    #     ax.legend(lns, labs, framealpha=1, bbox_to_anchor =(0.55, 1.4))

plt.show()

figure.set_facecolor("w")
plt.tight_layout()