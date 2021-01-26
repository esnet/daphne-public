import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Deeproute-stat-v0',
    entry_point='gym_deeproute_stat.envs:DeeprouteStatEnv',
)

