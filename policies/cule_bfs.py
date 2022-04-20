import torch
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

CROSSOVER_DICT = {'MsPacman': 1, 'Breakout': 2, 'Assault': 2, 'Krull': 2, 'Pong': 1, 'Boxing': 1, 'Asteroids': 1}


class CuleBFS():
    def __init__(self, step_env, tree_depth, gamma=0.99, compute_val_func=None, max_width=-1):
        if type(step_env) == DummyVecEnv:
            self.multiple_envs = True
            self.env_kwargs = step_env.envs[0].env_kwargs
            self.n_frame_stack = step_env.envs[0].n_frame_stack
        else:
            self.multiple_envs = False
            self.env_kwargs = step_env.env_kwargs
            self.n_frame_stack = step_env.n_frame_stack
        self.compute_val_func = compute_val_func
        self.crossover_level = 10
        for k, v in CROSSOVER_DICT.items():
            if k in self.env_kwargs['env_name']:
                self.crossover_level = v
                break
        self.clip_reward = step_env.clip_reward
        self.gamma = gamma
        self.max_depth = tree_depth
        cart = AtariRom(self.env_kwargs['env_name'])
        self.min_actions = cart.minimal_actions()
        self.min_actions_size = len(self.min_actions)
        num_envs = self.min_actions_size ** tree_depth if max_width == -1 \
            else min(max_width*self.min_actions_size, self.min_actions_size ** tree_depth)

        self.gpu_env = self.get_env(num_envs, device=torch.device("cuda", 0))
        if self.crossover_level == -1:
            self.cpu_env = self.gpu_env
        else:
            self.cpu_env = self.get_env(num_envs, device=torch.device("cpu"))
        if type(step_env) == DummyVecEnv:
            self.step_env = step_env.envs
        else:
            self.step_env = step_env.env

        self.num_leaves = num_envs
        self.gpu_actions = self.gpu_env.action_set
        self.cpu_actions = self.gpu_actions.to(self.cpu_env.device)

        self.device = self.gpu_env.device
        self.envs = [self.gpu_env]
        self.num_envs = 1
        self.trunc_count = 0
        self.max_width = max_width

    def get_env(self, num_envs, device):
        env = AtariEnv(num_envs=num_envs, device=device, action_set=self.min_actions, **self.env_kwargs)
        super(AtariEnv, env).reset(0)
        initial_steps_rand = 1
        env.reset(initial_steps=initial_steps_rand, verbose=True)
        env.set_size(1)
        # env.train()
        return env

    def bfs(self, state, tree_depth):
        # state_clone, rewards = self.bfs_orig(state, tree_depth)
        # state_clone_w, rewards_w, first_action = self.bfs_with_width(state, tree_depth)
        if self.max_width == -1:
            state_clone, rewards = self.bfs_orig(state, tree_depth)
            return state_clone, rewards, None
        return self.bfs_with_width(state, tree_depth)

    # state_clone, rewards = self.bfs_orig(state, tree_depth)
    # state_clone_w, rewards_w, first_action = self.bfs_with_width(state, tree_depth)
    # print((state_clone - state_clone_w).abs().sum(), (rewards - rewards_w).abs().sum())

    def bfs_orig(self, state, tree_depth):
        state_clone = state.clone().detach()

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
        # Create a default depth_env pointing to the CPU backend
        depth_env = cpu_env
        depth_actions_initial = self.cpu_actions
        num_envs = 1
        # TODO: Verify tree_depth=0, do we need to use step_env?
        relevant_env = depth_env if tree_depth > 0 else step_env
        for depth in range(tree_depth):
            # By level 3 there should be enough states to warrant moving to the GPU.
            # We do this by copying all of the relevant state information between the
            # backend GPU and CPU instances.
            if depth == self.crossover_level:
                self.copy_envs(cpu_env, gpu_env)
                depth_env = gpu_env
                relevant_env = depth_env if tree_depth > 0 else step_env
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
                # TODO: do we need this every frame, or just in the end?
                # Update the reward, done, and lives flags
                depth_env.get_data(depth_env.episodic_life, self.gamma ** depth, depth_env.done.data_ptr(),
                                   depth_env.rewards.data_ptr(), depth_env.lives.data_ptr())

                # To properly compute the output observations we need the last frame AND the second to last frame.
                # On the second to last step we need to update the frame buffers
                if frame == (depth_env.frameskip - 2):
                    depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                              depth_env.observations2[:num_envs].data_ptr())
                if frame == (depth_env.frameskip - 1):
                    depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                              depth_env.observations1[:num_envs].data_ptr())
            new_obs = torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs])
            # new_obs = new_obs / 255
            # import matplotlib.pyplot as plt
            # for i in range(4):
            #     plt.imshow(state_clone[i][0].cpu())
            #     plt.show()
            new_obs = new_obs.squeeze(dim=-1).unsqueeze(dim=1).to(self.device)
            state_clone = self.replicate_state(state_clone)
            state_clone = torch.cat((state_clone[:, 1: self.n_frame_stack, :, :], new_obs), dim=1)
            # obs = obs[:num_envs].to(gpu_env.device).permute((0, 3, 1, 2))
            # Waits for everything to finish running
            torch.cuda.synchronize()

        # Make sure all actions in the backend are completed
        if depth_env.is_cuda:
            depth_env.sync_this_stream()
            torch.cuda.current_stream().synchronize()

        # Form observations using max of last 2 frame_buffers
        # torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs], out=depth_env.observations1[:num_envs])

        # Waits for everything to finish running
        torch.cuda.synchronize()

        rewards = relevant_env.rewards[:num_envs].to(gpu_env.device)
        # TODO: Fix for depths > 1
        if self.clip_reward:
            rewards = torch.sign(rewards)
        cpu_env.set_size(1)
        gpu_env.set_size(1)
        return state_clone, rewards

    def bfs_with_width(self, state, tree_depth):
        state_clone = state.clone().detach()

        max_width = self.max_width

        cpu_env = self.cpu_env
        gpu_env = self.gpu_env
        step_env = self.step_env

        first_action = torch.arange(0, cpu_env.action_space.n, device=cpu_env.device).unsqueeze(1)

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
        # Create a default depth_env pointing to the CPU backend
        depth_env = cpu_env
        depth_actions_initial = self.cpu_actions
        num_envs = 1
        # TODO: Verify tree_depth=0, do we need to use step_env?
        relevant_env = depth_env if tree_depth > 0 else step_env
        for depth in range(tree_depth):
            # By level 3 there should be enough states to warrant moving to the GPU.
            # We do this by copying all of the relevant state information between the
            # backend GPU and CPU instances.
            if depth == self.crossover_level:
                self.copy_envs(cpu_env, gpu_env)
                depth_env = gpu_env
                relevant_env = depth_env if tree_depth > 0 else step_env
                depth_actions_initial = self.gpu_actions

            # Compute the number of environments at the current depth
            num_envs = self.min_actions_size ** (depth + 1) if max_width == -1 else \
                min(self.min_actions_size ** (depth + 1), max_width * self.min_actions_size)
            # depth_env.set_size(num_envs)
            self.new_expand(depth_env, num_envs)

            if depth != 0:
                first_action = first_action.repeat(1, cpu_env.action_space.n).view(-1, 1)

            depth_actions = depth_actions_initial.repeat(max_width)
            # Loop over the number of frameskips
            for frame in range(depth_env.frameskip):
                # Execute backend call to the C++ step function with environment data
                super(AtariEnv, depth_env).step(depth_env.fire_reset and depth_env.is_training, False,
                                                depth_actions.data_ptr(), 0, depth_env.done.data_ptr(), 0)
                # TODO: do we need this every frame, or just in the end?
                # Update the reward, done, and lives flags
                depth_env.get_data(depth_env.episodic_life, self.gamma ** depth, depth_env.done.data_ptr(),
                                   depth_env.rewards.data_ptr(), depth_env.lives.data_ptr())

                # To properly compute the output observations we need the last frame AND the second to last frame.
                # On the second to last step we need to update the frame buffers
                if frame == (depth_env.frameskip - 2):
                    depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                              depth_env.observations2[:num_envs].data_ptr())
                if frame == (depth_env.frameskip - 1):
                    depth_env.generate_frames(depth_env.rescale, False, depth_env.num_channels,
                                              depth_env.observations1[:num_envs].data_ptr())
            new_obs = torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs])
            # new_obs = new_obs / 255
            # import matplotlib.pyplot as plt
            # for i in range(4):
            #     plt.imshow(state_clone[i][0].cpu())
            #     plt.show()
            new_obs = new_obs.squeeze(dim=-1).unsqueeze(dim=1).to(self.device)
            state_clone = self.replicate_state(state_clone)
            state_clone = torch.cat((state_clone[:, 1: self.n_frame_stack, :, :], new_obs), dim=1)
            torch.cuda.synchronize()

            # TODO: make this work with estimate value instead of rewards
            if max_width != -1 and depth != tree_depth - 1 and num_envs > max_width:
                leaves_vals = self.compute_val_func(state_clone)[0].max(dim=1).values
                pi_logit = depth_env.rewards[:num_envs] + self.gamma**(depth + 1) * leaves_vals.to(depth_env.device)
                try:
                    top_indexes = torch.multinomial(torch.softmax(pi_logit, 0), max_width)
                except:
                    print("Bug in pi_logit", pi_logit)
                    top_indexes = torch.argsort(pi_logit, descending=True)[:max_width]
                first_action = first_action[top_indexes]
                state_clone = state_clone[top_indexes, :]
                depth_env.rewards[:max_width] = depth_env.rewards[top_indexes]
                depth_env.observations1[:max_width] = depth_env.observations1[top_indexes]
                depth_env.observations2[:max_width] = depth_env.observations2[top_indexes]
                depth_env.done[:max_width] = depth_env.done[top_indexes]
                depth_env.states[:max_width, :] = depth_env.states[top_indexes, :]
                depth_env.ram[:max_width] = depth_env.ram[top_indexes]
                depth_env.frame_states[:max_width] = depth_env.frame_states[top_indexes]
                depth_env.lives[:max_width] = depth_env.lives[top_indexes]
                depth_env.set_size(max_width)

        # Make sure all actions in the backend are completed
        if depth_env.is_cuda:
            depth_env.sync_this_stream()
            torch.cuda.current_stream().synchronize()

        # Form observations using max of last 2 frame_buffers
        # torch.max(depth_env.observations1[:num_envs], depth_env.observations2[:num_envs], out=depth_env.observations1[:num_envs])

        # Waits for everything to finish running
        torch.cuda.synchronize()

        rewards = relevant_env.rewards[:num_envs].to(gpu_env.device)
        # TODO: Fix for depths > 1
        if self.clip_reward:
            rewards = torch.sign(rewards)
        cpu_env.set_size(1)
        gpu_env.set_size(1)
        return state_clone, rewards, first_action

    def replicate_state(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        tmp = state.reshape(state.shape[0], -1)
        tmp = tmp.repeat(1, self.min_actions_size).view(-1, tmp.shape[1])
        return tmp.reshape(tmp.shape[0], *state.shape[1:])

    def copy_envs(self, source_env, target_env):
        target_env.set_size(source_env.size())
        target_env.states.copy_(source_env.states)
        target_env.ram.copy_(source_env.ram)
        target_env.rewards.copy_(source_env.rewards)
        target_env.done.copy_(source_env.done)
        target_env.frame_states.copy_(source_env.frame_states)
        torch.cuda.synchronize()
        target_env.update_frame_states()

    def new_expand(self, env, num_envs):
        orig_size = env.size()
        states = env.states[:env.size()].clone()
        ram = env.ram[:env.size()].clone()
        rewards = env.rewards[:env.size()].clone()
        done = env.done[:env.size()].clone()
        frame_states = env.frame_states[:env.size()].clone()
        lives = env.lives[:env.size()].clone()

        env.set_size(num_envs)

        #env_indices = torch.arange(num_envs, device=env.device) // int(num_envs / self.min_actions_size)
        env_indices = torch.arange(num_envs, device=env.device) // int(num_envs / orig_size)

        env.states[:env.size()] = states[env_indices]
        env.ram[:env.size()] = ram[env_indices]
        env.rewards[:env.size()] = rewards[env_indices]
        env.done[:env.size()] = done[env_indices]
        env.frame_states[:env.size()] = frame_states[env_indices]
        env.lives[:env.size()] = lives[env_indices]