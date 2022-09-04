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
def camel_to_snake(word):
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', word).lower()

def build_env(env_name):
    return AtariEnv(snake_to_camel(env_name) + 'NoFrameskip-v4', 1)

for i, env in enumerate(left_envs):
    print('{}/{} Trying to create {}'.format(i + 1, len(left_envs) , env))
    try:
      build_env(env)

    except Exception as exc:
        print(exc)
        bad_envs.add(env)

bad_envs.update({"bowling", "boxing", "double_dunk", "enduro", "fishing_derby",
            "freeway", "ice_hockey", "private_eye", "skiing", "star_gunner", "tennis"})

print('bad envs: {}'.format(bad_envs))
good_envs = all_envs - bad_envs
already_ran = {'AlienNoFrameskip-v4',
      'AmidarNoFrameskip-v4',
      'AssaultNoFrameskip-v4',
      'AsterixNoFrameskip-v4',
      'AsteroidsNoFrameskip-v4',
      'AtlantisNoFrameskip-v4',
      'BankHeistNoFrameskip-v4',
      'BattleZoneNoFrameskip-v4',
      'BerzerkNoFrameskip-v4',
      'BreakoutNoFrameskip-v4',
      'CentipedeNoFrameskip-v4',
      'ChopperCommandNoFrameskip-v4',
      'CrazyClimberNoFrameskip-v4',
      'FrostbiteNoFrameskip-v4',
      'GopherNoFrameskip-v4',
      'GravitarNoFrameskip-v4',
      'HeroNoFrameskip-v4',
      'KangarooNoFrameskip-v4',
      'KrullNoFrameskip-v4',
      'MsPacmanNoFrameskip-v4',
      'NameThisGameNoFrameskip-v4',
      'PhoenixNoFrameskip-v4',
      'PongNoFrameskip-v4',
      'RoadRunnerNoFrameskip-v4',
      'RobotankNoFrameskip-v4',
      'SeaquestNoFrameskip-v4',
      'SolarisNoFrameskip-v4',
      'SpaceInvadersNoFrameskip-v4',
      'TimePilotNoFrameskip-v4',
      'TutankhamNoFrameskip-v4',
      'VentureNoFrameskip-v4',
      'VideoPinballNoFrameskip-v4',
      'WizardOfWorNoFrameskip-v4',
      'YarsRevengeNoFrameskip-v4',
      'ZaxxonNoFrameskip-v4'}

for e in already_ran:
    good_envs.remove(camel_to_snake(e.replace('NoFrameskip-v4','')))

print('remaining good envs: {}'.format(good_envs))