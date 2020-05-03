import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Deeproute-v0',
    entry_point='gym_deeproute.envs:DeeprouteEnv',
)

