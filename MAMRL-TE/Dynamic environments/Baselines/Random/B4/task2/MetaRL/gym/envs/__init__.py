#!/usr/bin/env python3

from gym.envs.registration import register

register(
    'Deeproute-stat-v0',
    entry_point='MetaRL.gym.envs.deeproute.deeproute_stat_env:DeeprouteStatEnv',
)