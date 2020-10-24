import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id = 'cublirobot-v0',
    entry_point = 'cubli_robot.envs:Cubli_Env'
)