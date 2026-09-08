import argparse
import importlib.util
import os
from pathlib import Path

from play_umi_tac import PlayUMITac, _LOW_LEVEL_ROOT

from isaacgym_tactile import gymapi, gymtorch
from isaacgym_tactile.torch_utils import quat_apply, quat_rotate_inverse

import numpy as np
import torch
import cv2

_WBC_LOW_POLICY_PATH = _LOW_LEVEL_ROOT / "logs" / "whole_body_control" / "0601" / "ac_weights_45000.pt"
_ACTOR_CRITIC_PATH = (_LOW_LEVEL_ROOT / "legged_gym" / "envs" / "whole_body_control"
                      / "utils" / "actor_critic.py")


def _import_actor_critic():
    """Load ActorCritic straight from its file (pure torch) so the legged_gym
    package -- and with it the regular isaacgym -- is never imported."""
    spec = importlib.util.spec_from_file_location("wbc_actor_critic", _ACTOR_CRITIC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic


class PlayUMITacWBC(PlayUMITac):
    """PlayUMITac with the manip_loco movement control replaced by the
    whole_body_control low-level policy (mirrors WorldActionModelEnv.step())."""

    # Wine-bottle object, as in whole_body_control_pick_config.asset.target_file
    # and play_umi_tac_pick.py.
    OBJECT_URDF = (_LOW_LEVEL_ROOT / "resources" / "object" / "bottle" / "wine_bottle.urdf")
    OBJECT_INDENTER_LINK = "wine_bottle"

    # ── robot / control constants (mirrors whole_body_control_pick_config.py) ──
    DEFAULT_JOINT_ANGLES = {
        'FL_hip_joint': 0.1, 'FL_thigh_joint': 0.8, 'FL_calf_joint': -1.5,
        'RL_hip_joint': 0.1, 'RL_thigh_joint': 1.0, 'RL_calf_joint': -1.5,
        'FR_hip_joint': -0.1, 'FR_thigh_joint': 0.8, 'FR_calf_joint': -1.5,
        'RR_hip_joint': -0.1, 'RR_thigh_joint': 1.0, 'RR_calf_joint': -1.5,
        'Z_Joint1': 0.0, 'Z_Joint2': np.deg2rad(-70.0), 'Z_Joint3': np.deg2rad(60.0),
        'Z_Joint4': 0.0, 'Z_Joint5': np.deg2rad(16.0), 'Z_Joint6': 0.0,
        'Z_Joint_L': 0.0, 'Z_Joint_R': 0.0,
    }
    STIFFNESS = {'joint': 50.0, 'Z': 5.0}
    DAMPING = {'joint': 1.0, 'Z': 0.5}

    NUM_LOW_ACTIONS = 12
    LOW_ACTION_SCALE = 0.25
    HIP_SCALE_REDUCTION = 0.5
    HIP_INDICES = [0, 3, 6, 9]
    LAG_TIMESTEPS = 6                       # domain_rand.lag_timesteps
    CLIP_LOW_ACTIONS = 10.0                 # normalization.clip_actions

    LOW_LEVEL_NUM_OBS = 73
    LOW_LEVEL_OBS_HISTORY_LEN = 30          # 73 * 30 = 2190
    ARM_BASE_OFFSET = [-0.02, 0.0, 0.057]

    OBS_SCALES_LIN_VEL = 2.0
    OBS_SCALES_ANG_VEL = 0.25

    INIT_BASE_POS = [0.0, 0.0, 0.34]
    EPISODE_LENGTH_S = 20.0
    R_TERM, P_TERM, Z_TERM = 1.6, 1.6, 0.05

    LIN_VEL_X_CLIP = 0.3
    ANG_VEL_YAW_CLIP = 0.6

    # Body-frame Cartesian offsets from the reset EE pose (metres).
    # A one-second interpolation plus one-second hold keeps the motion gentle.
    EE_RANDOM_OFFSET_LOWER = [0.00, -0.03, -0.02]
    EE_RANDOM_OFFSET_UPPER = [0.10, 0.03, 0.05]
    EE_RANDOM_TRAJ_TIME_S = 1.0
    EE_RANDOM_HOLD_TIME_S = 1.0

    def __init__(self, low_policy_path, num_envs=1, device="cuda:0", headless=False,
                 sim_dt=0.005, decimation=4, viewer_width=1600, viewer_height=900,
                 record_wrist_camera=True, record_head_camera=True,
                 show_camera_markers=False):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.dt = sim_dt * decimation
        self.device_id = int(device.split(":")[1]) if "cuda" in device and ":" in device else 0

        self.record_wrist_camera = record_wrist_camera
        self.record_head_camera = record_head_camera
        self.show_camera_markers = show_camera_markers
        self.enable_cameras = record_wrist_camera or record_head_camera

        self.gym = gymapi.acquire_gym()
        self._create_sim()
        self._load_assets()
        self._create_envs()
        self._create_cameras()
        self._create_tactile_sensor()

        print("Preparing simulation ...")
        self.gym.prepare_sim(self.sim)

        self._init_buffers()
        self.tactile_sensor.post_load_after_prepare(self._rigid_body_state_full)

        self.viewer = None
        if not headless:
            self._create_viewer(viewer_width, viewer_height)

        print(f"Loading whole_body_control low-level policy from {low_policy_path}")
        self.low_level_policy = self._load_low_level_policy(low_policy_path)

        self.reset()
        print("PlayUMITacWBC ready")

    def _load_low_level_policy(self, low_policy_path):
        ActorCritic = _import_actor_critic()

        class actor_critic_config:
            init_noise_std = 1.0
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [512, 256, 128]
            activation = 'elu'
            adaptation_module_branch_hidden_dims = [256, 128]
            use_decoder = False

        policy = ActorCritic(
            self.LOW_LEVEL_NUM_OBS,
            2,  # num_privileged_obs
            self.LOW_LEVEL_NUM_OBS * self.LOW_LEVEL_OBS_HISTORY_LEN,
            self.NUM_LOW_ACTIONS,
            actor_critic_config(),
        )
        policy.load_state_dict(torch.load(low_policy_path, map_location=self.device))
        policy = policy.to(self.device)
        policy.eval()
        print("Low level pretrained policy loaded!")
        return policy

    # ─────────────────────────────────────────────────────────────────────
    # Buffers
    # ─────────────────────────────────────────────────────────────────────
    def _init_buffers(self):
        super()._init_buffers()
        self.arm_action_zero = torch.zeros(self.num_envs, self.NUM_ARM_DOF, device=self.device)
        self.arm_base_offset = torch.tensor(
            self.ARM_BASE_OFFSET, device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.last_low_actions = torch.zeros(self.num_envs, self.NUM_LOW_ACTIONS, device=self.device)
        self.last_last_low_actions = torch.zeros_like(self.last_low_actions)
        self.low_level_obs_history = torch.zeros(
            self.num_envs, self.LOW_LEVEL_NUM_OBS * self.LOW_LEVEL_OBS_HISTORY_LEN, device=self.device)
        self.lag_buffer = [torch.zeros(self.num_envs, self.NUM_TORQUES, device=self.device)
                           for _ in range(self.LAG_TIMESTEPS + 1)]

        self.ee_random_center_local = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_random_start_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_random_goal_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_random_timer = torch.zeros(self.num_envs, device=self.device)
        self.ee_random_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # whole_body_control resets joints to exact defaults
        # (manip_loco randomizes them by +-20%).
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        self.last_low_actions[env_ids] = 0.
        self.last_last_low_actions[env_ids] = 0.
        self.low_level_obs_history[env_ids] = 0.
        for lag in self.lag_buffer:
            lag[env_ids] = 0.
        self.gait_indices[env_ids] = 0.
        # The actual EE pose is reliable after the first simulated step. Until
        # then use the default joint targets, then make it the random-motion
        # center so a reset never produces an IK jump.
        self.ee_random_ready[env_ids] = False
        self.ee_random_timer[env_ids] = 0.

    def _resample_ee_random_goal(self, env_ids):
        if len(env_ids) == 0:
            return
        t = torch.clamp(self.ee_random_timer[env_ids] / self.EE_RANDOM_TRAJ_TIME_S, 0., 1.)
        current_offset = torch.lerp(
            self.ee_random_start_offset[env_ids], self.ee_random_goal_offset[env_ids], t[:, None])
        offset_lower = torch.tensor(self.EE_RANDOM_OFFSET_LOWER, device=self.device)
        offset_upper = torch.tensor(self.EE_RANDOM_OFFSET_UPPER, device=self.device)
        next_offset = offset_lower + torch.rand(len(env_ids), 3, device=self.device) * (
            offset_upper - offset_lower)
        self.ee_random_start_offset[env_ids] = current_offset
        self.ee_random_goal_offset[env_ids] = next_offset
        self.ee_random_timer[env_ids] = 0.

    # ─────────────────────────────────────────────────────────────────────
    # Control (ported from whole_body_control.py WorldActionModelEnv)
    # ─────────────────────────────────────────────────────────────────────
    def _compute_torques(self, low_actions):
        actions_scaled = low_actions * self.LOW_ACTION_SCALE
        actions_scaled[:, self.HIP_INDICES] *= self.HIP_SCALE_REDUCTION
        actions_scaled = torch.cat([actions_scaled, self.arm_action_zero], dim=-1)
        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        joint_pos_target = self.lag_buffer[0] + self.default_dof_pos_wo_gripper
        torques = self.p_gains * (joint_pos_target - self.dof_pos_wo_gripper) \
            - self.d_gains * self.dof_vel_wo_gripper
        torques[:, -self.NUM_ARM_DOF:] = 0
        torques = torch.cat([torques, self.gripper_torques_zero], dim=-1)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_low_level_observations(self):
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, self.arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.ee_pos - arm_base_pos)
        obs_buf = torch.cat((
            projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos_wo_gripper - self.default_dof_pos_wo_gripper) * self.OBS_SCALES_DOF_POS,
            self.dof_vel_wo_gripper * self.OBS_SCALES_DOF_VEL,
            ee_goal_local_cart,
            self.last_low_actions,
            self.last_last_low_actions,
            self.clock_inputs,
        ), dim=-1)
        self.low_level_obs_history = torch.cat(
            (self.low_level_obs_history[:, self.LOW_LEVEL_NUM_OBS:], obs_buf), dim=-1)

    def _update_curr_ee_goal(self):
        # Capture the reset pose only after physics has updated EE state. The
        # center and random offsets are body-frame quantities, so they travel
        # with the walking base instead of remaining fixed in the world frame.
        init_env_ids = (~self.ee_random_ready).nonzero(as_tuple=False).flatten()
        if len(init_env_ids) > 0:
            self.ee_random_center_local[init_env_ids] = quat_rotate_inverse(
                self.base_quat[init_env_ids],
                self.ee_pos[init_env_ids] - self.base_pos[init_env_ids],
            )
            self.ee_random_start_offset[init_env_ids] = 0.
            self.ee_random_goal_offset[init_env_ids] = 0.
            self.ee_random_ready[init_env_ids] = True
            self._resample_ee_random_goal(init_env_ids)

        t = torch.clamp(self.ee_random_timer / self.EE_RANDOM_TRAJ_TIME_S, 0., 1.)
        current_offset = torch.lerp(self.ee_random_start_offset, self.ee_random_goal_offset, t[:, None])
        self.curr_ee_goal_cart_world[:] = self.base_pos + quat_apply(
            self.base_yaw_quat, self.ee_random_center_local + current_offset)

        self.ee_random_timer += self.dt
        resample_ids = (self.ee_random_timer > (
            self.EE_RANDOM_TRAJ_TIME_S + self.EE_RANDOM_HOLD_TIME_S)).nonzero(as_tuple=False).flatten()
        self._resample_ee_random_goal(resample_ids)

    # ─────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────
    def compute_observations(self):
        # The manip_loco 71-dim proprio obs are unused here; the low-level
        # policy consumes low_level_obs_history built inside step().
        self.policy_obs = self.low_level_obs_history

    # ─────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────
    def step(self, actions=None):
        # High-level `actions` are ignored: like WorldActionModelEnv.step()
        # with zero top-level actions, movement is driven by self.commands
        # (x vel, yaw vel) through the pretrained low-level policy.
        self.commands[:, 0] = self.commands[:, 0].clamp(min=-1., max=1.)
        self.commands[:, 1] = 0.
        self.commands[:, 2] = self.commands[:, 2].clamp(min=-1., max=1.)

        self._compute_low_level_observations()
        with torch.no_grad():
            low_actions = self.low_level_policy.act_student(self.low_level_obs_history.detach())
        low_actions = torch.clip(low_actions, -self.CLIP_LOW_ACTIONS, self.CLIP_LOW_ACTIONS)
        self.last_last_low_actions[:] = self.last_low_actions[:]
        self.last_low_actions[:] = low_actions[:]
        self.actions = low_actions.clone()

        all_pos_targets = torch.zeros_like(self.dof_pos)
        arm_pos_targets = self.default_dof_pos_wo_gripper[-self.NUM_ARM_DOF:].repeat(self.num_envs, 1)
        ready_env_ids = self.ee_random_ready.nonzero(as_tuple=False).flatten()
        if len(ready_env_ids) > 0:
            dpos = self.curr_ee_goal_cart_world[ready_env_ids] - self.ee_pos[ready_env_ids]
            dpose = torch.cat((dpos, torch.zeros_like(dpos)), dim=-1).unsqueeze(-1)
            arm_pos_targets[ready_env_ids] = \
                self.dof_pos[ready_env_ids, -(self.NUM_ARM_DOF + self.NUM_GRIPPER_DOF):-self.NUM_GRIPPER_DOF] \
                + self._control_ik(dpose)
        all_pos_targets[:, -(self.NUM_ARM_DOF + self.NUM_GRIPPER_DOF):-self.NUM_GRIPPER_DOF] = arm_pos_targets
        all_pos_targets[:, -self.NUM_GRIPPER_DOF:] = self._get_gripper_position_targets()

        for _ in range(self.decimation):
            self.torques = self._compute_torques(self.last_low_actions)
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.render()
        self.post_physics_step()
        self.global_steps += 1
        return self.policy_obs


def main(env_cls=PlayUMITacWBC):
    parser = argparse.ArgumentParser(
        description="Replay whole_body_control low-level policy with tactile display")
    parser.add_argument("--low_policy_path", default=str(_WBC_LOW_POLICY_PATH),
                        help="Path to the whole_body_control ac_weights_*.pt checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--wrist_camera", dest="wrist_camera", action="store_true", default=True,
                        help="Gripper-mounted RGB camera (on by default)")
    parser.add_argument("--no_wrist_camera", dest="wrist_camera", action="store_false",
                        help="Disable the wrist camera")
    parser.add_argument("--head_camera", dest="head_camera", action="store_true", default=True,
                        help="Trunk-mounted RGB+depth camera (on by default)")
    parser.add_argument("--no_head_camera", dest="head_camera", action="store_false",
                        help="Disable the head camera")
    parser.add_argument("--camera_markers", dest="camera_markers", action="store_true", default=False,
                        help="Draw wireframe-sphere + forward-axis camera pose markers (off by default)")
    parser.add_argument("--no_camera_markers", dest="camera_markers", action="store_false",
                        help="Hide the camera pose markers")
    args = parser.parse_args()

    if not os.path.exists(args.low_policy_path):
        raise FileNotFoundError(f"No whole_body_control low-level policy at {args.low_policy_path}")

    env = env_cls(
        low_policy_path=args.low_policy_path,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        record_wrist_camera=args.wrist_camera,
        record_head_camera=args.head_camera,
        show_camera_markers=args.camera_markers,
    )

    step = 0
    try:
        while True:
            # The base velocity command and the small IK EE offset are both
            # resampled internally; no external policy actions are needed.
            env.step()

            if not args.headless:
                env._visualize_tactile()
                if env.enable_cameras:
                    env._visualize_cameras()
                cv2.waitKey(1)

            step += 1
            if args.steps is not None and step >= args.steps:
                break
    except (KeyboardInterrupt, SystemExit) as exc:
        print("Stopping:", exc)
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
