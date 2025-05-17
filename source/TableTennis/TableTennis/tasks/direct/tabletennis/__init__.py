# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Tabletennis-Direct-v0",
    entry_point=f"{__name__}.tabletennis_env:TabletennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tabletennis_env_cfg:TabletennisEnvCfg",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_simple.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_auto.yaml",
    },
)

gym.register(
    id="Template-Tabletennis-Direct-Camera-v0",
    entry_point=f"{__name__}.tabletennis_camera_env:TabletennisCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tabletennis_camera_env_cfg:TabletennisCameraEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_simple_ppo_cfg.yaml",
    },
)