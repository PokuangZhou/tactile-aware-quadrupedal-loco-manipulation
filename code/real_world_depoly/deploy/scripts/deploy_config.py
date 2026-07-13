
CFG = {
    "control": {
        "decimation":          4,      # control decimation, freq = 1/(decimation*sim_dt) #50hz
        "action_scale":        0.25,   # action scale
        "hip_scale_reduction": 1.0,    # refer to UMI on leg, 1.0 = no reduction
        "stiffness":           {"joint": 50.0}, #{"joint": 40.0},
        "damping":             {"joint": 1.0},
    },
    "sim": {
        "dt": 0.005,   # sim dt, policy freq = 1/(decimation*dt) = 50Hz
    },
    "env": {
        "num_observation_history": 10,
        "num_proprio":             71,
        "num_priv":                18,
        "num_envs":                1,
        "num_actions":             12,  # dog only, arm zeroed out
    },  
    "init_state": {
        "default_joint_angles": {
            "FL_hip_joint":    0.1,
            "FL_thigh_joint":  0.8,
            "FL_calf_joint":  -1.5,
            "FR_hip_joint":   -0.1,
            "FR_thigh_joint":  0.8,
            "FR_calf_joint":  -1.5,
            "RL_hip_joint":    0.1,
            "RL_thigh_joint":  0.8,
            "RL_calf_joint":  -1.5,
            "RR_hip_joint":   -0.1,
            "RR_thigh_joint":  0.8,
            "RR_calf_joint":  -1.5,
        }
    },
    "normalization": {
        "clip_actions":      10.0,
        "clip_observations": 100.0,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    },
}