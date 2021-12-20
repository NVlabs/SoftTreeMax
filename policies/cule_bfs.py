import torch
import numpy as np
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom
import time

RAND_FIRE_LIST = ['Breakout']
CROSSOVER_DICT = {'MsPacman': 1, 'Breakout': 2, 'Assault': 2, 'Krull': 2, 'Pong': 1, 'Boxing': 1, 'Asteroids': 1}
OP_FACTOR_DICT = {'Video': [1.1] * 4, 'Space': [1.15] * 4, 'Breakout': [1.125, 1.2, 1.125, 1.075],
                  'Asteroids': [1.0] * 4,
                  'Frostbite': [1.1] * 4, 'Beam': [1.15, 1.025, 1.025]}


class CuleBFS():
    def __init__(self, env_name, tree_depth, env_kwargs, gamma=0.99, verbose=False, ale_start_steps=1,
                 ignore_value_function=False, step_env=None, args=None):
        self.env_kwargs = env_kwargs
        self.crossover_level = 1
        for k, v in CROSSOVER_DICT.items():
            if k in env_name:
                self.crossover_level = v
                break
        self.op_scale_factor = args.op_scale_factor
        self.op_scale_factor_per_depth = [self.op_scale_factor] * 4
        if not args.override_default_op_factor:
            for k, v in OP_FACTOR_DICT.items():
                if k in env_name:
                    self.op_scale_factor = v[tree_depth - 1]
                    self.op_scale_factor_per_depth = v
                    break
        self.args = args
        self.verbose = verbose
        self.print_times = False
        self.ale_start_steps = ale_start_steps
        self.gamma = gamma
        self.max_depth = tree_depth
        self.env_name = env_name
        self.ignore_value_function = ignore_value_function

        cart = AtariRom(env_name)
        self.min_actions = cart.minimal_actions()
        self.min_actions_size = len(self.min_actions)
        num_envs = self.min_actions_size ** tree_depth

        self.gpu_env = self.get_env(num_envs, device=torch.device("cuda", 0))
        if self.crossover_level == -1:
            self.cpu_env = self.gpu_env
        else:
            self.cpu_env = self.get_env(num_envs, device=torch.device("cpu"))
        self.step_env = step_env

        self.num_leaves = num_envs
        self.gpu_actions = self.gpu_env.action_set
        self.cpu_actions = self.gpu_actions.to(self.cpu_env.device)

        self.device = self.gpu_env.device
        self.envs = [self.gpu_env]
        self.num_envs = 1
        self.trunc_count = 0

    def get_env(self, num_envs, device):
        env = AtariEnv(self.env_name, num_envs, device=device, action_set=self.min_actions, **self.env_kwargs)
        super(AtariEnv, env).reset(0)
        initial_steps_rand = 1
        env.reset(initial_steps=initial_steps_rand, verbose=self.verbose)
        # env.train()
        return env

    def bfs(self, state, q_net, support, args, fire_pressed=[False], max_cut_time=[0]):
        state_clone = state.clone().detach()
        max_depth = args.tree_depth
        fire_env = len([e for e in RAND_FIRE_LIST if e in self.env_name]) > 0
        if fire_env and np.random.rand() < 0.01:
            # make sure 'FIRE' is pressed often enough to launch ball after life loss
            # return torch.tensor([1], device=self.device), torch.tensor(0, device=self.device)
            fire_pressed[0] = True
            return 1
        print_times = self.print_times
        cpu_env = self.cpu_env
        gpu_env = self.gpu_env
        step_env = self.step_env

        # Set device environment root state before calling step function
        cpu_env.states[0] = step_env.states[0]
        cpu_env.ram[0] = step_env.ram[0]
        cpu_env.frame_states[0] = step_env.frame_states[0]

        # Zero out all buffers before calling any environment functions
        cpu_env.rewards.zero_()
        cpu_env.observations1.zero_()
        cpu_env.observations2.zero_()
        cpu_env.done.zero_()

        # Make sure all actions in the backend are completed
        # Be careful making calls to pytorch functions between cule synchronization calls
        if gpu_env.is_cuda:
            gpu_env.sync_other_stream()

        if print_times:
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()

        # Create a default depth_env pointing to the CPU backend
        depth_env = cpu_env
        depth_actions_initial = self.cpu_actions
        num_envs = 1
        relevant_env = depth_env if max_depth > 0 else step_env
        for depth in range(max_depth):
            if print_times:
                depth_start = torch.cuda.Event(enable_timing=True)
                depth_end = torch.cuda.Event(enable_timing=True)
                depth_start.record()

            # By level 3 there should be enough states to warrant moving to the GPU.
            # We do this by copying all of the relevant state information between the
            # backend GPU and CPU instances.
            if depth == self.crossover_level:
                self.copy_envs(cpu_env, gpu_env)
                depth_env = gpu_env
                relevant_env = depth_env if max_depth > 0 else step_env
                depth_actions_initial = self.gpu_actions

            # Compute the number of environments at the current depth
            num_envs = self.min_actions_size ** (depth + 1)
            # depth_env.set_size(num_envs)
            depth_env.expand(num_envs)

            depth_actions = depth_actions_initial.repeat(self.min_actions_size ** depth)
            # Loop over the number of frameskips
            for frame in range(depth_env.frameskip):
                # Execute backend call to the C++ step function with environment data
                super(AtariEnv, depth_env).step(depth_env.fire_reset and depth_env.is_training, False,
                                                depth_actions.data_ptr(), 0, depth_env.done.data_ptr(), 0)
                # Update the reward, done, and lives flags
                depth_env.get_data(depth_env.episodic_life, self.gamma ** depth, depth_env.done.data_ptr(),
                                   depth_env.rewards.data_ptr(), depth_env.lives.data_ptr())

                # To properly compute the output observations we need the last frame AND the second to last frame.
                # On the second to last step we need to update the frame buffers
                if not self.ignore_value_function:
                    if frame == (depth_env.frameskip - 2):
                        depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                                  depth_env.observations2[:num_envs].data_ptr())
                    if frame == (depth_env.frameskip - 1):
                        depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                                  depth_env.observations1[:num_envs].data_ptr())
            new_obs = torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs])
            new_obs = new_obs / 255
            # import matplotlib.pyplot as plt
            # for i in range(4):
            #     plt.imshow(state_clone[i][0].cpu())
            #     plt.show()
            new_obs = new_obs.squeeze(dim=-1).unsqueeze(dim=1).to(self.device)
            state_clone = self.replicate_state(state_clone)

            state_clone = torch.cat((state_clone[:, 1: args.history_length, :, :], new_obs), dim=1)
            # obs = obs[:num_envs].to(gpu_env.device).permute((0, 3, 1, 2))
            if print_times:
                depth_end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            if print_times:
                depth_runtime = depth_start.elapsed_time(depth_end)
                print('Level {} with {} environments: {:4.4f} (ms)'.format(depth, num_envs, depth_runtime))
            if depth < max_depth - 1:
                cut_time_start = time.time()
                max_cut_condition = self.use_max_diff_cut and \
                                    self.compute_max_cut_condition(gpu_env, num_envs, state_clone, q_net,
                                                                   depth=depth, env=relevant_env, support=support)
                cut_time_end = time.time()
                max_cut_time[0] += cut_time_end - cut_time_start
                if max_cut_condition:
                    max_depth = depth + 1
                    torch.cuda.synchronize()
                    break
        # relevant_env = depth_env if max_depth > 0 else step_env
        self.mean_depth_vec.append(max_depth)
        # positive_reward = num_envs > relevant_env.rewards[:num_envs].to(gpu_env.device).sum() > 0 or True
        d0_values = self.compute_value(state, q_net, support)
        d0_act = d0_values.argmax(1).item()
        if max_depth == 0:  # or not positive_reward:
            return d0_act

        # Make sure all actions in the backend are completed
        if depth_env.is_cuda:
            depth_env.sync_this_stream()
            torch.cuda.current_stream().synchronize()

        # Form observations using max of last 2 frame_buffers
        # torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs], out=depth_env.observations1[:num_envs])
        if print_times:
            total_end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        if print_times:
            total_runtime = total_start.elapsed_time(total_end)
            print('Total expansion time: {:4.4f} (ms)'.format(total_runtime))
            value_start = torch.cuda.Event(enable_timing=True)
            value_end = torch.cuda.Event(enable_timing=True)
            value_start.record()

        rewards = self.compute_rewards(gpu_env, num_envs, state_clone, q_net, depth=max_depth, env=relevant_env,
                                       support=support)
        # Break ties with small noise
        rewards += torch.normal(mean=torch.zeros_like(rewards), std=1e-5)
        best_value = rewards.max()
        size_subtree = self.min_actions_size ** (max_depth - 1)
        d0_act_idx = slice(size_subtree * d0_act, size_subtree * (d0_act + 1))
        if self.args.multiplicative_op_factor:
            rewards[d0_act_idx] *= self.op_scale_factor_per_depth[max_depth - 1]
        else:
            rewards[d0_act_idx] += (self.op_scale_factor_per_depth[max_depth - 1] - 1) * rewards.mean()
            # delta_half = abs(d0_values - rewards_depth1)[0] / 2
            # sigma_o = delta_half[d0_act]
            # sigma_e = (delta_half.sum() - sigma_o) / (len(delta_half) - 1)
            # bonus = self.gamma ** max_depth * np.sqrt(2 * np.log(self.min_actions_size)) * (sigma_e * np.sqrt(max_depth) - sigma_o * np.sqrt(max_depth - 1))
            # rewards[d0_act_idx] += bonus
        best_action = rewards.argmax() // depth_env.action_space.n ** (max_depth - 1)
        self.trunc_count += int(best_action != d0_act)

        cpu_env.set_size(1)
        gpu_env.set_size(1)
        if False and print_times:  # currently cancelled due to crash
            torch.cuda.synchronize()
            value_end.record()
            value_runtime = value_start.elapsed_time(value_end)
            print('Total value computation time: {:4.4f} (ms)'.format(value_runtime))

        return best_action.unsqueeze(-1)

    def replicate_state(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        tmp = state.reshape(state.shape[0], -1)
        tmp = tmp.repeat(1, self.min_actions_size).view(-1, tmp.shape[1])
        return tmp.reshape(tmp.shape[0], *state.shape[1:])

    def compute_rewards(self, gpu_env, num_envs, obs, q_net, depth, env, support):
        rewards = env.rewards[:num_envs].to(gpu_env.device)
        done = env.done[:num_envs].to(gpu_env.device)
        if (~done).any() and not self.ignore_value_function:
            not_done_value = self.compute_value(obs[~done, :], q_net, support).max(1)[0]
            rewards[~done] += (self.gamma ** depth) * not_done_value
        return rewards

    def copy_envs(self, source_env, target_env):
        print_times = self.print_times
        if print_times:
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_end = torch.cuda.Event(enable_timing=True)
        target_env.set_size(source_env.size())

        if print_times:
            copy_start.record()
        target_env.states.copy_(source_env.states)
        target_env.ram.copy_(source_env.ram)
        target_env.rewards.copy_(source_env.rewards)
        target_env.done.copy_(source_env.done)
        target_env.frame_states.copy_(source_env.frame_states)
        if print_times:
            copy_end.record()

        torch.cuda.synchronize()
        target_env.update_frame_states()
        if print_times:
            depth_copytime = copy_start.elapsed_time(copy_end)
            print('Depth copy time: {:4.4f} (ms)'.format(depth_copytime))

    def compute_value(self, state, q_net, support):
        with torch.no_grad():
            if len(state.shape) == 3:
                state = state.unsqueeze(dim=0)
            # return (q_net(state.unsqueeze(0)) * support).sum(2).argmax(1).item()
            if state.shape[0] == 2744:
                stds = [
                    ((q_net(state)[0, j, :] * support ** 2).sum() - (q_net(state)[0, j, :] * support).sum() ** 2).item()
                    for j in range(14)]
                if max(stds) - min(stds) > 0.1:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    plt.plot(support.cpu(), q_net(state)[0, np.argmax(stds), :].cpu())
                    plt.plot(support.cpu(), q_net(state)[0, np.argmin(stds), :].cpu())
                    plt.show()
                    fig = plt.figure()
                    for i in range(14):
                        plt.plot(support.cpu(), q_net(state)[0, i, :].cpu())
                    plt.show()
            return (q_net(state) * support).sum(2)

