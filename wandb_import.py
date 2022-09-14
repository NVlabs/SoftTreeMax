import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

FROM_CSV = False
CSV_PATH = 'pgtree_results.h5'
store = pd.HDFStore(CSV_PATH)

if not FROM_CSV:
    api = wandb.Api()
    project_name = 'pg-tree'
    runs = api.runs("nvr-israel/{}".format(project_name))
    summary_list = []
    config_list = []
    name_list = []
    steps_list = []
    rewards_list = []
    sweep_ids = ['e2oawpk0', 'n5sealqt', 'iqpkpwcr', 'k9cwlb2n']
    for run in runs:
        if run.sweep is not None and run.sweep.id in sweep_ids:
            h_df = run.scan_history() # can either use run.history() or, instead, scan_history() and build list of dicts via [r for r in h_df]
            hist_dicts = [r for r in h_df]
            if len(hist_dicts) > 0: # len(h_df) > 0:
                # run.summary are the output key/values like accuracy.
                # We call ._json_dict to omit large files
                summary_list.append(run.summary._json_dict)

                # run.config is the input metrics.
                # We remove special values that start with _.
                config = {k:v for k,v in run.config.items() if not k.startswith('_')}
                config['sweep_id'] = run.sweep.id
                config_list.append(config)

                # run.name is the name of the run.
                name_list.append(run.name)

                # history_list.append(h_df.filter(['_step', 'reward']).to_records(index=False)) #df = df.astype('object')

                # steps_list.append(h_df.filter(['_step']))
                # rewards_list.append(list(h_df.filter(['reward']).reward))
                env_name = config_list[0]['env_name']
                if 'MsPacman' in env_name:
                    rew_str = 'episodic_reward'
                elif 'maze' in env_name:
                    rew_str = 'tot_reward'
                else:
                    rew_str = 'reward'
                steps = [e['_step'] for e in hist_dicts]
                rewards = [e[rew_str] for e in hist_dicts]
                steps_list.append(steps)
                rewards_list.append(rewards)
                # rewards_list.append(list(h_df.filter(['reward']).reward))

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    # history_df = pd.DataFrame.from_records(history_list)
    reward_df = pd.DataFrame({'reward_vec': [np.nan] * len(name_df)})
    reward_df['reward_vec'] = reward_df['reward_vec'].astype('object')
    for i_delay, l in enumerate(rewards_list):
        reward_df.reward_vec[i_delay] = l
    step_df = pd.DataFrame({'step_vec': [np.nan] * len(name_df)})
    step_df['step_vec'] = step_df['step_vec'].astype('object')
    for i_step, s in enumerate(steps_list):
        step_df.step_vec[i_step] = s

    all_df = pd.concat([name_df, config_df, summary_df, reward_df, step_df], axis=1)
else:
    # all_df = pd.read_csv(CSV_PATH)
    all_df = store['df']


convergence_plots = True
if convergence_plots and FROM_CSV:
    fontsize = 14
    all_df = all_df[all_df.name != 'bumbling-sweep-15']
    all_df = all_df[all_df.name != 'honest-sweep-26']
    n_rolling_mean = 0 #50
    colors = {'Augmented-Q': 'blue', 'Delayed-Q': 'green', 'Oblivious-Q': 'red'}
    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w), 'valid') / w
    def moving_average(x, alpha):
        out = [x[0]]
        c = 1
        for i in range(len(x) - 1):
            new_val = alpha * out[-1] + (1 - alpha) * x[c]
            out.append(new_val)
            c += 1
        return np.asarray(out)

    # figure, axes = plt.subplots(3, 4)
    figure, big_axes = plt.subplots(figsize=(15.0, 15.0), nrows=3, ncols=1, sharey=True)

    CSV_PATH_LIST = ['maze_convergence.h5', 'noisy_cartpole_convergence.h5', 'pacman_convergence.h5']
    for i_env, csv_path in enumerate(CSV_PATH_LIST):
        curr_store = pd.HDFStore(csv_path)
        all_df = curr_store['df']

        env_name_orig = all_df.env_name.iloc[0]
        delay_str = 'delay_value'
        agent_str = 'agent_type'
        LABEL_DICT = {'augmented': 'Augmented-Q', 'oblivious': 'Oblivious-Q', 'delayed': 'Delayed-Q'}
        maze_delay_sweeps = {0: 'e2oawpk0', 5: 'n5sealqt', 15: 'iqpkpwcr', 25: 'k9cwlb2n'}

        if 'maze' in env_name_orig:
            env_name = 'Maze'
            delay_str = 'delay'
            agent_str = 'exp_label'
            LABEL_DICT = {'augmentedQ': 'Augmented-Q', 'obliviousQ': 'Oblivious-Q', 'modQ': 'Delayed-Q'}
        elif 'MsPacman' in env_name_orig:
            env_name = 'MsPacman'
        else:
            env_name = 'Noisy CartPole'

        big_axes[i_env].set_title(env_name, fontsize=19)
        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_axes[i_env].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_axes[i_env]._frameon = False

        df_delays = all_df.groupby([delay_str])
        # figure, axes = plt.subplots(1, len(df_delays.groups))
        for i_delay, delay_value in enumerate(df_delays.groups):
            if 'maze' in env_name_orig:
                all_df_filtered = all_df[all_df.sweep_id == maze_delay_sweeps[delay_value]]
                df_delays = all_df_filtered.groupby([delay_str])
            if delay_value > 15:
                alpha_smoothing = 0.99
            else:
                alpha_smoothing = 0.98
            is_first = i_delay == 0
            df_env = df_delays.get_group(delay_value)[['name', agent_str, 'seed', 'reward_vec', 'step_vec']]
            df_agents = df_env.groupby([agent_str])
            for i_agent, agent_type in enumerate(df_agents.groups):  # iterate on agents
                if agent_type == 'rnn':
                    break
                df_agent = df_agents.get_group(agent_type)
                min_y_len = min(df_agent.reward_vec.str.len()) - n_rolling_mean
                smallest_x = min([l[-1] for l in df_agent.step_vec])
                # max_y_len = max(df_agent.reward_vec.str.len()) - n_rolling_mean
                y_vals_vec = None
                longest_x_vals = []
                for i_run in range(df_agent.shape[0]):  # iterate on seeds
                    plot_vals = df_agent.iloc[i_run].reward_vec
                    x_vals = np.asarray(df_agent.iloc[i_run].step_vec)
                    x_vals = x_vals [x_vals <= smallest_x]
                    # if len(x_vals) > len(longest_x_vals):
                    #     longest_x_vals = x_vals
                    # x_vals = [e[0] for e in plot_vals][n_rolling_mean - 1:]
                    # y_vals = moving_average(plot_vals, n_rolling_mean)[:min_y_len]
                    y_vals = moving_average(plot_vals, alpha_smoothing)[:len(x_vals)]
                    # joint_seq = [(x, y) for (x, y) in zip(x_vals, y_vals)]
                    # ts = traces.TimeSeries(data=joint_seq)
                    # regularized = ts.moving_average(start=joint_seq[0][0], sampling_period=1, placement='left')
                    # print('done')
                    # x_vals = [e[0] for e in regularized]; y_vals = [e[1] for e in regularized]
                    # padding_len = max_y_len - len(y_vals)
                    # if padding_len > 0:
                    #     padding = np.empty((padding_len,)); padding[:] = np.nan
                    #     y_vals = np.concatenate((y_vals, padding))
                    f = interp1d(x_vals, y_vals)
                    x_vals = np.linspace(np.min(x_vals), np.max(x_vals), round(smallest_x / 3))
                    y_vals = f(x_vals)
                    if y_vals_vec is None:
                        y_vals_vec = y_vals
                    else:
                        y_vals_vec = np.vstack((y_vals_vec, y_vals))
                label = LABEL_DICT[agent_type]
                color = colors[label]
                i_subplot = 1 + i_env * 4 + i_delay
                ax = figure.add_subplot(3, 4, i_subplot)

                if len(y_vals_vec.shape) > 1:
                    y_mean = np.nanmean(y_vals_vec, axis=0)
                    # y_mean = np.median(y_vals_vec, axis=0)
                    y_std = np.nanstd(y_vals_vec, axis=0)
                    # y_std = y_vals_vec.nanstd(axis=0)
                    under_line = (y_mean - y_std)
                    over_line = (y_mean + y_std)
                    ax.plot(x_vals, y_mean, linewidth=1, label=label, color=color)  # mean curve.
                    ax.fill_between(x_vals, under_line, over_line, color='b', alpha=.1)
                else:
                    ax.plot(x_vals, y_vals_vec, linewidth=1, label=label, color=color)  # mean curve.
                # axes[i_delay].set_ylim(bottom=0)
                ax.set_title('Delay m={}'.format(delay_value), fontsize=fontsize)
                if i_delay == 0:
                    ax.set_ylabel('Reward', fontsize=fontsize)
                ax.set_xlabel('Steps', fontsize=fontsize)
                ax.grid('on')
                if 'maze' not in env_name_orig:
                    import matplotlib.ticker as ticker
                    ax.xaxis.set_major_formatter(ticker.EngFormatter())
                    if 'MsPacman' in env_name_orig:
                        ax.set_xlim(0, 1000000)
                    else:
                        ax.set_xlim(0, 250000)
            if i_delay == 0:
                lines_labels = ax.get_legend_handles_labels()
                lines, labels = lines_labels[0], lines_labels[1]
                # legend1 = figure.legend(lines, labels, loc='upper left', fontsize=fontsize)
                legend1 = figure.legend(lines, labels, loc=(0.01, 0.89), fontsize=fontsize)
                legend2 = figure.legend(lines, labels, loc=(0.01, 0.58), fontsize=fontsize)
                figure.legend(lines, labels, loc=(0.01, 0.26), fontsize=fontsize)
                figure.gca().add_artist(legend1)
                figure.gca().add_artist(legend2)

    figure.set_facecolor('w')
    plt.tight_layout()