from torchcule.atari import Env as AtariEnv

all_envs = {'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist',
     'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede',
     'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk', 'elevator_action', 'enduro',
     'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape',
     'kaboom', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix',
     'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
     'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture',
     'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'}

bad_envs = {'defender'}
failed_envs = {}
left_envs = all_envs - bad_envs


def snake_to_camel(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))

def build_env(env_name):
    return AtariEnv(snake_to_camel(env_name) + 'NoFrameskip-v4', 1)

for i, env in enumerate(left_envs):
    print('{}/{} Trying to create {}'.format(i + 1, len(left_envs) , env))
    try:
      build_env(env)

    except Exception as exc:
        print(exc)
        bad_envs.add(env)

print('bad envs: {}'.format(bad_envs))
print('good envs: {}'.format(all_envs - bad_envs))

