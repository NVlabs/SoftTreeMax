import torch
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom

CROSSOVER_DICT = {'MsPacman': 1, 'Breakout': 2, 'Assault': 2, 'Krull': 2, 'Pong': 1, 'Boxing': 1, 'Asteroids': 1}


class CuleBFS():
    def __init__(self, env_name, tree_depth, env_kwargs, gamma=0.99, verbose=False, ale_start_steps=1,
                 ignore_value_function=False, step_env=None, args=None):
        self.env_kwargs = env_kwargs
        self.crossover_level = 1
        for k, v in CROSSOVER_DICT.items():
            if k in env_name:
                self.crossover_level = v
                break
        self.args = args
        self.verbose = verbose
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

    def bfs(self, state, tree_depth, n_frame_stack):
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
            state_clone = torch.cat((state_clone[:, 1: n_frame_stack, :, :], new_obs), dim=1)
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
        cpu_env.set_size(1)
        gpu_env.set_size(1)
        return state_clone, rewards

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

