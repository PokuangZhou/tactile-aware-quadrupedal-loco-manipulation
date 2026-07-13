import time
import lcm
import numpy as np
import torch

from go2_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    """
    LCM deployment agent for the Leandown project.

    Main differences from Go2 Gym:
    - num_obs: 42 → 71 (num_proprio)
    - history_len: 30 → 10
    - Adds joint and foot reindexing.
    - Adds gait-clock computation.
    - Uses a completely different observation structure.
    """
    
    def __init__(self, cfg, se, command_profile):
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.timestep = 0

        # ============ Leandown configuration ============
        # Observation dimensions.
        self.num_proprio = 71  # Single-step proprioceptive dimension, including the gait clock.
        self.num_priv = 18     # Privileged-information dimension.
        self.history_len = 10  # History length.
        
        # Total observation dimension = num_proprio + num_priv + num_proprio * history_len.
        self.num_obs = self.num_proprio * (self.history_len + 1) + self.num_priv  # 799
        
        self.num_envs = 1
        self.num_actions = 12 #18  # The model outputs 18 dimensions (12 leg + 6 arm).
        self.num_leg_actions = 12  # Use only the 12 leg actions.
        self.num_commands = 3  # Velocity commands (vx, vy, yaw).
        self.device = 'cpu'

        # ============ Observation scale factors ============
        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]
        
        # Command scaling for the three velocity commands.
        self.commands_scale = np.array([
            self.obs_scales["lin_vel"],
            self.obs_scales["lin_vel"], 
            self.obs_scales["ang_vel"]
        ])

        # ============ Joint configuration ============
        # Go2 leg-joint names in the original URDF order.
        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]        

        # ===== default joint angles (safe) =====
        default_angles = self.cfg.get("init_state", {}).get("default_joint_angles", {})

        # Fallback: use the specified values.
        FALLBACK_LEG_DEFAULTS = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
        }

        missing = [n for n in joint_names if n not in default_angles]
        if missing:
            print(f"[WARN] cfg.init_state.default_joint_angles missing: {missing}")
            print("[WARN] Using FALLBACK_LEG_DEFAULTS for missing joints.")
            self.default_dof_pos_leg = np.array(
                [default_angles.get(n, FALLBACK_LEG_DEFAULTS[n]) for n in joint_names],
                dtype=np.float32
            )
        else:
            self.default_dof_pos_leg = np.array([default_angles[n] for n in joint_names], dtype=np.float32)

        
        # Default arm-joint angles for the six joints.
        arm_joint_names = ["Z_Joint1", "Z_Joint2", "Z_Joint3", "Z_Joint4", "Z_Joint5", "Z_Joint6"]
        self.default_dof_pos_arm = np.array([0.0, 1.57, 0.0, 0.0, 0.0, 0.0])  # From lean_down_config.py.
        
        # Complete set of 18 default joint angles.
        self.default_dof_pos = np.concatenate([self.default_dof_pos_leg, self.default_dof_pos_arm])

        # ============ PD gains ============
        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")
        print(f"d_gains: {self.d_gains}")

        # ============ State buffers ============
        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(self.num_leg_actions)  # Store only leg actions.
        # self.smooth_actions = torch.zeros(self.num_leg_actions) ## 🌀
        # self.action_smooth_alpha = 0.6  # tune this: higher = less filtering ## 🌀
        self.last_actions = torch.zeros(self.num_leg_actions)
        
        # IMU data.
        self.gravity_vector = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.body_linear_vel = np.zeros(3)
        
        # Joint data for the 12 leg joints only.
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        
        # Foot contacts.
        self.foot_contacts = np.zeros(4)
        self.contact_state = np.ones(4)
        
        # Control targets.
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)

        self.joint_idxs = self.se.joint_idxs

        # ============ Gait clock ============
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)
        
        # Fixed gait parameters from lean_down_config.py.
        self.gait_frequency = 2.0  # Hz
        self.gait_phase = 0.5
        self.gait_offset = 0.0
        self.gait_bound = 0.0
        self.gait_duration = 0.5

        # ============ Fixed end-effector goal ============
        self.ee_goal_local_cart = np.zeros(3)  # End-effector target position.
        self.ee_goal_orientation = np.zeros(3)  # End-effector target orientation.

        # ============ Fixed privileged information ============
        # priv_buf = [mass_params(5), friction(1), motor_strength(12)]
        self.priv_fixed = np.zeros(self.num_priv)
        self.priv_fixed[5] = 1.0  # Set the friction coefficient to 1.0.
        # motor_strength - 1 = 0 for standard strength.

        self.is_currently_probing = False

        print(f"============ Leandown LCM Agent 初始化完成 ============")
        print(f"num_proprio: {self.num_proprio}")
        print(f"num_priv: {self.num_priv}")
        print(f"history_len: {self.history_len}")
        print(f"total obs_buf dim: {self.num_obs}")

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def _reindex_dof(self, vec):
        """
        Reorder joints from [FL, FR, RL, RR] to [FR, FL, RR, RL].

        This corresponds to the _reindex_all function used during Leandown training.
        """
        # Input order: [FL(0,1,2), FR(3,4,5), RL(6,7,8), RR(9,10,11)].
        # Output order: [FR(3,4,5), FL(0,1,2), RR(9,10,11), RL(6,7,8)].
        return vec[[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _reindex_dof_inverse(self, vec):
        """
        Restore joint order from [FR, FL, RR, RL] to [FL, FR, RL, RR].

        This permutation is self-inverse, so applying it twice restores the original order.
        """
        return vec[[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _reindex_feet(self, vec):
        """
        Reorder feet from [FL, FR, RL, RR] to [FR, FL, RR, RL].

        This corresponds to the _reindex_feet function used during Leandown training.
        """
        return vec[[1, 0, 3, 2]]

    def _get_body_orientation(self):
        """
        Compute roll and pitch from the gravity vector.

        Returns: [roll, pitch] (2D).
        """
        # Get roll and pitch from the IMU; the state estimator already provides RPY.
        rpy = self.se.get_rpy()
        return np.array([rpy[0], rpy[1]])

    def _compute_gait_clock(self):
        """
        Compute gait-clock inputs.

        Based on the _step_contact_targets function in lean_down.py.
        """
        # Update the gait index.
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * self.gait_frequency, 1.0
        )
        
        # Reset the gait index when the command is zero.
        if np.abs(self.commands[0, 0]) < 0.1 and np.abs(self.commands[0, 2]) < 0.1:
            self.gait_indices[:] = 0.0

        # Compute phases for all four feet.
        phases = self.gait_phase
        offsets = self.gait_offset
        bounds = self.gait_bound
        durations = self.gait_duration

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases
        ]

        # Apply stance/swing phase adjustments.
        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) >= durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (0.5 / (1 - durations))

        # Compute sinusoidal clock inputs.
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

    def get_obs(self):
        """
        Build the Leandown observation vector.

        obs_prop (71D) = [
            body_orientation (2),     # roll, pitch
            base_ang_vel (3),         # Angular velocity.
            dof_pos_leg (12),         # Reordered leg-joint positions.
            dof_pos_arm (6),          # Arm-joint positions, fixed at zero.
            dof_vel_leg (12),         # Reordered leg-joint velocities.
            dof_vel_arm (6),          # Arm-joint velocities, fixed at zero.
            last_actions (12),        # Leg actions from the previous step.
            foot_contacts (4),        # Reordered foot contacts.
            commands (3),             # Velocity commands.
            ee_goal_pos (3),          # End-effector target position, fixed at zero.
            ee_goal_orn (3),          # End-effector target orientation, fixed at zero.
            gait_indices (1),         # Gait index.
            clock_inputs (4),         # Gait clock.
        ]
        
        Returns: obs_prop (71D NumPy array).
        """
        # Get IMU data.
        self.gravity_vector = self.se.get_gravity_vector()
        self.body_angular_vel = self.se.get_body_angular_vel()

        # print("\n===== 👀[Leandown] OBS DEBUG =====")
        # print(f"[DEBUG] gravity_vector: {self.gravity_vector}")
        # print(f"[DEBUG] body_angular_vel: {self.body_angular_vel}")

        # Replace NaN values in body_angular_vel with zero.
        if np.isnan(self.body_angular_vel).any():
            print("[WARN] body_angular_vel contains nan, replacing with 0")
            self.body_angular_vel = np.nan_to_num(self.body_angular_vel, nan=0.0)
        
        # Get commands.
        cmds, reset_timer = self.command_profile.get_command(
            self.timestep * self.dt, probe=self.is_currently_probing
        )
        self.commands[:, :] = cmds[:self.num_commands]
        if reset_timer:
            self.reset_gait_indices()

        # Get joint data.
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        
        # Get foot contacts.
        self.contact_state = self.se.get_contact_state()
        # Convert foot forces to binary contact states using a threshold.
        self.foot_contacts = (self.contact_state > 20).astype(np.float32)

        # Compute the gait clock.
        self._compute_gait_clock()

        # ============ Build obs_prop (71D) ============
        
        # 1. body_orientation (2D): roll, pitch.
        body_orientation = self._get_body_orientation()
        
        # 2. base_ang_vel (3D).
        ang_vel_scaled = self.body_angular_vel * self.obs_scales["ang_vel"]
        
        # 3. dof_pos_leg (12D): reordered leg-joint positions.
        dof_pos_leg_raw = (self.dof_pos - self.default_dof_pos_leg) * self.obs_scales["dof_pos"]
        dof_pos_leg = self._reindex_dof(dof_pos_leg_raw)
        
        # 4. dof_pos_arm (6D): fixed at zero.
        dof_pos_arm = np.zeros(6)
        
        # 5. dof_vel_leg (12D): reordered leg-joint velocities.
        dof_vel_leg_raw = self.dof_vel * self.obs_scales["dof_vel"]
        dof_vel_leg = self._reindex_dof(dof_vel_leg_raw)
        
        # 6. dof_vel_arm (6D): fixed at zero.
        dof_vel_arm = np.zeros(6)
        
        # 7. last_actions (12D): reordered leg actions from the previous step.
        last_actions_clipped = torch.clip(
            self.actions, 
            -self.cfg["normalization"]["clip_actions"],
            self.cfg["normalization"]["clip_actions"]
        ).cpu().detach().numpy()

        # ===== Replace NaN values =====
        if np.isnan(last_actions_clipped).any():
            print("[WARN] last_actions contains nan, replacing with 0")
            last_actions_clipped = np.nan_to_num(last_actions_clipped, nan=0.0)
        
        # 8. foot_contacts (4D): reordered foot contacts.
        foot_contacts_reindexed = self._reindex_feet(self.foot_contacts)
        
        # 9. commands (3D).
        commands_scaled = self.commands.flatten() * self.commands_scale
        
        # 10. ee_goal_pos (3D): fixed at zero.
        ee_goal_pos = self.ee_goal_local_cart
        
        # 11. ee_goal_orn (3D): fixed at zero.
        ee_goal_orn = self.ee_goal_orientation
        
        # 12. gait_indices (1D).
        gait_indices = self.gait_indices.numpy()
        
        # 13. clock_inputs (4D).
        clock_inputs = self.clock_inputs.numpy().flatten()

        # Concatenate all observations.
        obs_prop = np.concatenate([
            body_orientation,           # 2
            ang_vel_scaled,             # 3
            dof_pos_leg,                # 12
            dof_pos_arm,                # 6
            dof_vel_leg,                # 12
            dof_vel_arm,                # 6
            last_actions_clipped,       # 12
            foot_contacts_reindexed,    # 4
            commands_scaled,            # 3
            ee_goal_pos,                # 3
            ee_goal_orn,                # 3
            gait_indices,               # 1
            clock_inputs,               # 4
        ])  # 71 dimensions in total.
        # ===== Debug output =====
        if self.timestep % 200 == 0:
            print("\n===== OBS DEBUG =====")
            print(f"① 原始 dof_pos (FL,FR,RL,RR): {self.dof_pos}")
            print(f"② default_dof_pos_leg: {self.default_dof_pos_leg}")
            print(f"③ dof_pos - default (原始): {self.dof_pos - self.default_dof_pos_leg}")
            print(f"④ 重排序后 dof_pos_leg (FR,FL,RR,RL): {dof_pos_leg}")
            print("======")

        return torch.tensor(obs_prop, device=self.device).float()

    def get_privileged_observations(self):
        """Return the fixed privileged information."""
        return torch.tensor(self.priv_fixed, device=self.device).float()

    def publish_action(self, action, hard_reset=False):
        """
        Publish actions to the robot.
        
        Args:
            action: Model output actions (18D or 12D).
            hard_reset: Whether to perform a hard reset.
        """
        command_for_robot = pd_tau_targets_lcmt()
        
        # Use only the leg actions (the first 12 dimensions).
        if action.shape[-1] >= 12:
            leg_action = action[0, :12].detach().cpu().numpy()
        else:
            leg_action = action[0, :].detach().cpu().numpy()
        
        # Restore order from [FR, FL, RR, RL] to [FL, FR, RL, RR].
        leg_action_robot = self._reindex_dof_inverse(leg_action)
        
        # Compute target joint positions.
        self.joint_pos_target = leg_action_robot * self.cfg["control"]["action_scale"]
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target = self.joint_pos_target + self.default_dof_pos_leg
        
        # Apply the joint-index mapping.
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        command_for_robot.kp = self.p_gains
        command_for_robot.kd = self.d_gains
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        # Compute torques for logging.
        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + \
                       (self.joint_vel_target - self.dof_vel) * self.d_gains
        
        # ===== Debug output =====
        if self.timestep % 200 == 0:
            print("\n===== ACTION DEBUG =====")
            print(f"⑤ 神经网络输出 action (FR,FL,RR,RL): {leg_action}")
            print(f"⑥ 逆重排序后 action (FL,FR,RL,RR): {leg_action_robot}")
            print(f"⑦ action * action_scale: {leg_action_robot * self.cfg['control']['action_scale']}")
            print(f"⑧ joint_pos_target (发给机器人): {self.joint_pos_target}")
            print("======" )
            
        # Publish the LCM message.
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        """Reset the agent state."""
        self.actions = torch.zeros(self.num_leg_actions)
        # self.smooth_actions = torch.zeros(self.num_leg_actions) ## 🌀        
        self.last_actions = torch.zeros(self.num_leg_actions)
        self.time = time.time()
        self._last_send_time = time.time()
        self.timestep = 0
        self.reset_gait_indices()
        return self.get_obs()

    def reset_gait_indices(self):
        """Reset the gait index."""
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, hard_reset=False):
        """
        Execute one step.
        
        Args:
            actions: Model output action tensor.
            hard_reset: Whether to perform a hard reset.
            
        Returns:
            obs: Observation.
            None: Placeholder.
            None: Placeholder.
            infos: Information dictionary.
        """
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions.clone()
        
        # Store only the leg actions (the first 12 dimensions) and reorder them.
        if actions.shape[-1] >= 12:
            leg_actions = actions[0, :12]
        else:
            leg_actions = actions[0, :]
        self.actions = torch.clip(leg_actions, -clip_actions, clip_actions) ##  🌀
        # clipped = torch.clip(leg_actions, -clip_actions, clip_actions)
        # self.smooth_actions = self.action_smooth_alpha * clipped + (1 - self.action_smooth_alpha) * self.smooth_actions
        # self.actions = self.smooth_actions
        
        # Publish actions.
        self.publish_action(actions, hard_reset=hard_reset)
        
        # Wait for the control period.
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        self.time = time.time()
        if self.timestep % 100 == 0:
            print(f'send joint action frq🔍: {1 / (time.time() - self._last_send_time):.1f} Hz')
        self._last_send_time = time.time()
        
        # Get observations.
        obs = self.get_obs()

        # Build the information dictionary.
        infos = {
            "joint_pos": self.dof_pos[np.newaxis, :],
            "joint_vel": self.dof_vel[np.newaxis, :],
            "joint_pos_target": self.joint_pos_target[np.newaxis, :],
            "joint_vel_target": self.joint_vel_target[np.newaxis, :],
            "body_linear_vel": self.body_linear_vel[np.newaxis, :],
            "body_angular_vel": self.body_angular_vel[np.newaxis, :],
            "contact_state": self.contact_state[np.newaxis, :],
            "clock_inputs": self.clock_inputs.numpy()[np.newaxis, :],
            "body_linear_vel_cmd": self.commands[:, 0:2],
            "body_angular_vel_cmd": self.commands[:, 2:],
            "privileged_obs": None,
        }

        self.timestep += 1
        return obs, None, None, infos


class HistoryWrapper:
    """
    Historical-observation wrapper.

    Extend a single-step observation into an observation buffer containing history.

    Differences from Go2 Gym:
    - obs_history_length: 30 → 10
    - Adds priv_buf to obs_buf.
    - Uses the obs_buf structure [obs_prop, priv_buf, obs_history].
    """
    
    def __init__(self, env):
        self.env = env

        # Leandown configuration.
        self.num_proprio = 71  # Single-step observation dimension.
        self.num_priv = 18     # Privileged-information dimension.
        self.history_len = 10  # History length.
        
        # History buffer: (num_envs, history_len, num_proprio).
        self.obs_history_buf = torch.zeros(
            self.env.num_envs, self.history_len, self.num_proprio,
            dtype=torch.float, device=self.env.device, requires_grad=False
        )
        
        # Fixed privileged information.
        self.priv_buf = torch.tensor(
            self.env.priv_fixed, dtype=torch.float, device=self.env.device
        ).unsqueeze(0)
        
        # Total observation dimension.
        self.num_obs = self.num_proprio + self.num_priv + self.num_proprio * self.history_len  # 799
        
        self.num_privileged_obs = self.env.num_priv

        print(f"============ History Wrapper 初始化完成 ============")
        print(f"num_proprio: {self.num_proprio}")
        print(f"num_priv: {self.num_priv}")
        print(f"history_len: {self.history_len}")
        print(f"obs_buf dim: {self.num_obs}")

    def _update_history(self, obs_prop):
        """
        Update the history buffer.
        
        Args:
            obs_prop: Current single-step observation (num_proprio dimensions).
        """
        # Roll the history: discard the oldest entry and append the newest.
        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, 1:, :],
            obs_prop.unsqueeze(0).unsqueeze(1)
        ], dim=1)

    def _build_obs_buf(self, obs_prop):
        """
        Build the complete obs_buf.

        Structure: [obs_prop(71), priv_buf(18), obs_history(710)].
        Total: 799 dimensions.
        """
        obs_history_flat = self.obs_history_buf.view(self.env.num_envs, -1)  # (1, 710)
        
        obs_buf = torch.cat([
            obs_prop.unsqueeze(0),      # (1, 71)
            self.priv_buf,              # (1, 18)
            obs_history_flat            # (1, 710)
        ], dim=-1)  # (1, 799)
        
        return obs_buf

    def step(self, action):
        """Execute one step and return an observation containing history."""
        obs_prop, rew, done, info = self.env.step(action)
        
        # Update history.
        self._update_history(obs_prop)
        
        # Build obs_buf.
        obs_buf = self._build_obs_buf(obs_prop)
        
        return {
            'obs': obs_prop.unsqueeze(0),      # Current observation (1, 71).
            'obs_buf': obs_buf,                 # Complete observation (1, 799).
            'privileged_obs': self.priv_buf,   # Privileged information (1, 18).
            'obs_history': self.obs_history_buf.clone()  # History buffer (1, 10, 71).
        }, rew, done, info

    def get_observations(self):
        """Get the current observation."""
        obs_prop = self.env.get_obs()
        self._update_history(obs_prop)
        obs_buf = self._build_obs_buf(obs_prop)
        
        return {
            'obs': obs_prop.unsqueeze(0),
            'obs_buf': obs_buf,
            'privileged_obs': self.priv_buf,
            'obs_history': self.obs_history_buf.clone()
        }

    def get_obs(self):
        """Get the current observation (alias)."""
        return self.get_observations()

    def reset_idx(self, env_ids):
        """Reset history for the specified environments."""
        ret = self.env.reset_idx(env_ids)
        self.obs_history_buf[env_ids, :, :] = 0
        return ret

    def reset(self):
        """Reset all environments."""
        obs_prop = self.env.reset()
        
        # Clear history.
        self.obs_history_buf[:, :, :] = 0
        
        # Fill history with the current observation.
        for _ in range(self.history_len):
            self._update_history(obs_prop)
        
        obs_buf = self._build_obs_buf(obs_prop)
        
        return {
            'obs': obs_prop.unsqueeze(0),
            'obs_buf': obs_buf,
            'privileged_obs': self.priv_buf,
            'obs_history': self.obs_history_buf.clone()
        }

    def __getattr__(self, name):
        """Delegate other attributes to the underlying environment."""
        return getattr(self.env, name)
