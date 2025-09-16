# Import existing environments
from PysoroGym.envs.push_box_env import PushBoxEnv
# Add the new circular environment
from PysoroGym.envs.push_box_env import CircularPushBoxEnv

# Make them available when importing from PysoroGym.envs
__all__ = [
    'PushBoxEnv',
    'CircularPushBoxEnv'
]