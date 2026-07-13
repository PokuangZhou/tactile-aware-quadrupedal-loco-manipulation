#!/usr/bin/env python3
# python play_umi_tac.py --exptid 0323 --checkpoint 45000
#
import argparse
import math
import os
import sys
from pathlib import Path


def _ensure_libpython_visible():
    """Restart with the active conda env's lib dir in LD_LIBRARY_PATH."""
    if os.name != "posix":
        return

    major = sys.version_info.major
    minor = sys.version_info.minor
    lib_dir = Path(sys.prefix) / "lib"
    libpython = lib_dir / f"libpython{major}.{minor}.so.1.0"

    if not libpython.exists():
        return

    current_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if str(lib_dir) in current_paths:
        return

    if os.environ.get("_PLAY_UMI_TAC_LD_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(lib_dir), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    env["_PLAY_UMI_TAC_LD_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


_ensure_libpython_visible()

from isaacgym_tactile import gymapi, gymtorch, gymutil
from isaacgym_tactile.torch_utils import (
    to_torch, quat_apply, quat_conjugate, quat_mul, quat_from_euler_xyz,
    quat_rotate_inverse, get_euler_xyz, get_axis_params, torch_rand_float,
)

import numpy as np
import torch
import cv2
from typing import Optional

from tactile_sensor import TactileSensor

_THIS_FILE = Path(__file__).resolve()
_LOW_LEVEL_ROOT = _THIS_FILE.parents[1]                       # .../low-level
_RESOURCES_ROOT = _LOW_LEVEL_ROOT / "resources"
_DOG_URDF = _RESOURCES_ROOT / "go2d1_tactile_array" / "urdf" / "go2d1_umi_tac.urdf"
_BOX_URDF = _RESOURCES_ROOT / "object" / "box" / "backpack.urdf"
_BOX_INDENTER_LINK = "base_link"


# ─────────────────────────────────────────────────────────────────────────
# Pure-math helpers ported verbatim from manip_loco.py (no isaacgym import)
# ─────────────────────────────────────────────────────────────────────────
def euler_from_quat(quat):
    roll, pitch, yaw = get_euler_xyz(quat)
    roll = (roll + np.pi) % (2 * np.pi) - np.pi
    pitch = (pitch + np.pi) % (2 * np.pi) - np.pi
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return roll, pitch, yaw


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone()
    quat_yaw[:, 0] = 0.
    quat_yaw[:, 1] = 0.
    quat_yaw = quat_yaw / torch.norm(quat_yaw, dim=-1, keepdim=True).clamp(min=1e-9)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def sphere2cart(sphere_coords):
    # type: (Tensor) -> Tensor
    l = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = l * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = l * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = l * torch.sin(pitch)
    return cart_coords


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class PlayUMITac:
    """Replays a trained manip_loco policy on isaacgym_tactile + shows tactile heatmaps."""

    OBJECT_URDF = _BOX_URDF
    OBJECT_INDENTER_LINK = _BOX_INDENTER_LINK
    OBJECT_FIX_BASE_LINK = True
    OBJECT_MASS = None

    # ── robot / control constants (mirrors manip_loco_config.py) ──────────
    NUM_LEG_DOF = 12
    NUM_ARM_DOF = 6
    NUM_GRIPPER_DOF = 2
    NUM_DOFS = NUM_LEG_DOF + NUM_ARM_DOF + NUM_GRIPPER_DOF       # 20
    NUM_TORQUES = NUM_LEG_DOF + NUM_ARM_DOF                      # 18
    NUM_ACTIONS = NUM_TORQUES                                    # 18
    ACTION_DELAY = 3

    DEFAULT_JOINT_ANGLES = {
        'FL_hip_joint': 0.1, 'FL_thigh_joint': 0.8, 'FL_calf_joint': -1.5,
        'RL_hip_joint': 0.1, 'RL_thigh_joint': 0.8, 'RL_calf_joint': -1.5,
        'FR_hip_joint': -0.1, 'FR_thigh_joint': 0.8, 'FR_calf_joint': -1.5,
        'RR_hip_joint': -0.1, 'RR_thigh_joint': 0.8, 'RR_calf_joint': -1.5,
        'Z_Joint1': 0.0, 'Z_Joint2': -1.57, 'Z_Joint3': 1.57,
        'Z_Joint4': 0.0, 'Z_Joint5': 0.0, 'Z_Joint6': 0.0,
        'Z_Joint_L': 0.0, 'Z_Joint_R': 0.0,
    }
    STIFFNESS = {'joint': 40.0, 'Z': 5.0}
    DAMPING = {'joint': 1.0, 'Z': 0.5}
    ACTION_SCALE = [0.4, 0.45, 0.45] * 4 + [2.1, 0.6, 0.6, 0, 0, 0]
    ARM_DOF_STIFFNESS = 400.0
    ARM_DOF_DAMPING = 40.0

    INIT_BASE_POS = [0.0, 0.0, 0.32]
    RAND_YAW_RANGE = np.pi / 2
    ORIGIN_PERTURB_RANGE = 0.5
    INIT_VEL_PERTURB_RANGE = 0.1

    BOX_OFFSET = [2.0, 0.0, 0.2]

    R_TERM, P_TERM, Z_TERM = 0.8, 0.8, 0.1
    EPISODE_LENGTH_S = 10.0

    OBS_SCALES_ANG_VEL = 1.0
    OBS_SCALES_DOF_POS = 1.0
    OBS_SCALES_DOF_VEL = 0.05
    OBS_SCALES_LIN_VEL = 1.0
    NUM_PROPRIO_BASE = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3       # 66
    HISTORY_LEN = 10
    OBSERVE_GAIT_COMMANDS = True
    NUM_PROPRIO = NUM_PROPRIO_BASE + (5 if OBSERVE_GAIT_COMMANDS else 0)  # 71

    COMMAND_RESAMPLING_TIME = 3.0
    LIN_VEL_X_RANGE = [-0.8, 0.8]
    ANG_VEL_YAW_RANGE = [-1.0, 1.0]
    LIN_VEL_X_CLIP = 0.2
    ANG_VEL_YAW_CLIP = 0.4

    GAIT_FREQUENCY = 2.0
    GAIT_KAPPA = 0.07  # unused directly (desired_contact_states not fed to obs), kept for fidelity

    GOAL_TRAJ_TIME = [3.0, 4.5]
    GOAL_HOLD_TIME = [0.5, 2.0]
    GOAL_POS_L = [0.3, 0.4]
    GOAL_POS_P = [0.0, np.pi / 3]
    GOAL_POS_Y = [-1.5, 1.5]
    GOAL_DELTA_ORN_R = [-0.5, 0.5]
    GOAL_DELTA_ORN_P = [-0.5, 0.5]
    GOAL_DELTA_ORN_Y = [-0.5, 0.5]
    GOAL_INIT_POS_START = [0.3, np.pi / 8, 0.0]
    GOAL_INIT_POS_END = [0.4, 0.0, 0.0]
    GOAL_COLLISION_UPPER = [0.6, 0.4, 0.4]
    GOAL_COLLISION_LOWER = [-0.0, -0.4, -0.1]
    GOAL_UNDERGROUND_LIMIT = -0.3
    GOAL_NUM_COLLISION_SAMPLES = 10
    GOAL_ARM_INDUCED_PITCH = 0.0
    GOAL_SPHERE_CENTER_OFFSET = [0.03, 0.0, 0.37]   # x_offset, y_offset, z_invariant_offset

    CLIP_ACTIONS = 100.0
    CLIP_OBSERVATIONS = 100.0

    # ── tactile sensor grid (same as teleop_umi_ee.py) ─────────────────────
    TACTILE_NUM_ROWS = 12
    TACTILE_NUM_COLUMNS = 32
    TACTILE_POINT_DISTANCE = 0.002

    # ── cameras (mirrors manip_loco_nav_config defaults) ───────────────────
    WRIST_CAMERA_BETWEEN_LINKS = ("Empty_Link_L", "Empty_Link_R")  # midpoint = between fingers
    WRIST_CAMERA_ORIENT_LINK = "Empty_Link6"                       # palm; +X = grasp approach
    WRIST_CAMERA_POS = [-0.02, 0.02, 0.07]           # palm-frame offset: forward to the pad/grasp plane
    WRIST_CAMERA_QUAT = [0.0, 0.1305262, 0.0, 0.9914449]  # +15 deg around palm-frame Y
    WRIST_CAMERA_WIDTH = 160
    WRIST_CAMERA_HEIGHT = 120
    WRIST_CAMERA_FOV = 75.0

    HEAD_CAMERA_LINK = "base"
    HEAD_CAMERA_POS = [0.34, 0.0, 0.01]            # local offset in base frame
    HEAD_CAMERA_QUAT = [0.0, 0.0, 0.0, 1.0]        # local orientation (xyzw)
    HEAD_CAMERA_WIDTH = 160
    HEAD_CAMERA_HEIGHT = 120
    HEAD_CAMERA_FOV = 75.0
    HEAD_DEPTH_NEAR = 0.1                           # for depth visualisation only
    HEAD_DEPTH_FAR = 3.0

    def __init__(self, jit_policy_path, num_envs=1, device="cuda:0", headless=False,
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
        # Camera sensors need a graphics device; force one on even when headless.
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

        print(f"Loading jit policy from {jit_policy_path}")
        self.policy = torch.jit.load(jit_policy_path, map_location=self.device)

        self.reset()
        print("PlayUMITac ready")

    # ─────────────────────────────────────────────────────────────────────
    # Simulation / asset / env creation
    # ─────────────────────────────────────────────────────────────────────
    def _create_sim(self):
        p = gymapi.SimParams()
        p.dt = self.sim_dt
        p.substeps = 1
        p.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        p.up_axis = gymapi.UP_AXIS_Z
        p.use_gpu_pipeline = True

        p.physx.use_gpu = True
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_position_iterations = 8
        p.physx.num_velocity_iterations = 1
        p.physx.contact_offset = 0.002
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 1000.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

        graphics_device = 0 if (not self.headless or self.enable_cameras) else -1
        self.sim = self.gym.create_sim(self.device_id, graphics_device, gymapi.SIM_PHYSX, p)
        if self.sim is None:
            raise RuntimeError("Failed to create sim")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane)

    def _load_assets(self):
        dog_opt = gymapi.AssetOptions()
        dog_opt.default_dof_drive_mode = 3  # effort
        dog_opt.collapse_fixed_joints = True
        dog_opt.replace_cylinder_with_capsule = True
        dog_opt.flip_visual_attachments = True
        dog_opt.fix_base_link = False
        dog_opt.use_mesh_materials = True
        self.dog_asset = self.gym.load_asset(self.sim, str(_DOG_URDF.parent), _DOG_URDF.name, dog_opt)

        self.num_dofs = self.gym.get_asset_dof_count(self.dog_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.dog_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.dog_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(self.dog_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(self.dog_asset)
        self.gripper_idx = self.body_names_to_idx["end_effector"]

        self.feet_names = [s for s in self.body_names if "foot" in s]

        dof_props = self.gym.get_asset_dof_properties(self.dog_asset)
        dof_props['driveMode'][self.NUM_LEG_DOF:].fill(gymapi.DOF_MODE_POS)
        dof_props['stiffness'][self.NUM_LEG_DOF:].fill(self.ARM_DOF_STIFFNESS)
        dof_props['damping'][self.NUM_LEG_DOF:].fill(self.ARM_DOF_DAMPING)
        self.dog_dof_props = dof_props

        box_opt = gymapi.AssetOptions()
        box_opt.density = 1000
        box_opt.fix_base_link = self.OBJECT_FIX_BASE_LINK
        box_opt.disable_gravity = False
        self.box_asset = self.gym.load_asset(
            self.sim,
            str(self.OBJECT_URDF.parent),
            self.OBJECT_URDF.name,
            box_opt,
        )

        print(f"num_dofs={self.num_dofs} num_bodies={self.num_bodies} feet={self.feet_names} "
              f"gripper_idx={self.gripper_idx}")

    def _create_envs(self):
        num_per_row = max(1, int(np.sqrt(self.num_envs)))
        spacing = 3.0
        env_lo = gymapi.Vec3(0., 0., 0.)
        env_hi = gymapi.Vec3(0., 0., 0.)

        self.envs = []
        self.dog_handles = []
        self.box_handles = []
        self.actor_handles_list = []   # for TactileSensor: [{"robot_dog": h, "box": h}, ...]

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lo, env_hi, num_per_row)
            self.envs.append(env)

            row, col = divmod(i, num_per_row)
            origin = gymapi.Vec3(row * spacing, col * spacing, 0.0)

            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(
                origin.x + self.INIT_BASE_POS[0],
                origin.y + self.INIT_BASE_POS[1],
                self.INIT_BASE_POS[2],
            )
            start_pose.r = gymapi.Quat.from_euler_zyx(0., 0., self.RAND_YAW_RANGE * np.random.uniform(-1, 1))

            dog_handle = self.gym.create_actor(env, self.dog_asset, start_pose, "robot_dog", i, 0, 0)
            self.gym.set_actor_dof_properties(env, dog_handle, self.dog_dof_props)
            self.dog_handles.append(dog_handle)

            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(start_pose.p.x + self.BOX_OFFSET[0],
                                      start_pose.p.y + self.BOX_OFFSET[1],
                                      self.BOX_OFFSET[2])
            box_handle = self.gym.create_actor(env, self.box_asset, box_pose, "box", i, 0, 0)
            if self.OBJECT_MASS is not None:
                body_props = self.gym.get_actor_rigid_body_properties(env, box_handle)
                body_props[0].mass = self.OBJECT_MASS
                self.gym.set_actor_rigid_body_properties(
                    env, box_handle, body_props, recomputeInertia=True)
            self.box_handles.append(box_handle)

            self.actor_handles_list.append({"robot_dog": dog_handle, "box": box_handle})

        self.feet_indices = torch.tensor(
            [self.gym.find_actor_rigid_body_handle(self.envs[0], self.dog_handles[0], n) for n in self.feet_names],
            dtype=torch.long, device=self.device)

        assert len(self.feet_names) == 4, f"expected 4 feet, got {self.feet_names}"

    def _create_tactile_sensor(self):
        tactile_configs = [
            {
                "elastomer_actor_name": "robot_dog",
                "elastomer_urdf_path": str(_DOG_URDF),
                "elastomer_link_name": "elastomer_left",
                "indenter_actor_name": "box",
                "indenter_urdf_path": str(self.OBJECT_URDF),
                "indenter_link_name": self.OBJECT_INDENTER_LINK,
            },
            {
                "elastomer_actor_name": "robot_dog",
                "elastomer_urdf_path": str(_DOG_URDF),
                "elastomer_link_name": "elastomer_right",
                "indenter_actor_name": "box",
                "indenter_urdf_path": str(self.OBJECT_URDF),
                "indenter_link_name": self.OBJECT_INDENTER_LINK,
            },
        ]
        self.tactile_sensor = TactileSensor(
            gym=self.gym, sim=self.sim, envs=self.envs,
            actor_handles=self.actor_handles_list, num_envs=self.num_envs, device=self.device,
            tactile_configs=tactile_configs,
            tactile_num_rows=self.TACTILE_NUM_ROWS,
            tactile_num_columns=self.TACTILE_NUM_COLUMNS,
            tactile_point_distance=self.TACTILE_POINT_DISTANCE,
        )
        self.tactile_sensor.post_load_before_prepare()

    def _create_viewer(self, width, height):
        vp = gymapi.CameraProperties()
        vp.width = width
        vp.height = height
        self.viewer = self.gym.create_viewer(self.sim, vp)
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
        self.gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(2.5, 2.5, 1.8), gymapi.Vec3(0.0, 0.0, 0.3))
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")

    # ─────────────────────────────────────────────────────────────────────
    # Cameras (ported from manip_loco_nav.py, adapted to isaacgym_tactile)
    # ─────────────────────────────────────────────────────────────────────
    def _create_cameras(self):
        if not self.enable_cameras:
            return
        if self.record_wrist_camera:
            self._create_wrist_camera_sensors()
        if self.record_head_camera:
            self._create_head_camera_sensors()

    def _make_camera_props(self, width, height, fov):
        props = gymapi.CameraProperties()
        props.width = width
        props.height = height
        props.horizontal_fov = fov
        props.enable_tensors = False   # CPU readback via get_camera_image
        return props

    def _find_camera_body_handle(self, env_id, link_name, camera_name):
        body_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[env_id], self.dog_handles[env_id], link_name)
        if body_handle < 0:
            raise RuntimeError(
                f"Could not find {camera_name} camera link '{link_name}'. "
                f"Available robot links: {', '.join(self.body_names)}")
        return body_handle

    def _create_wrist_camera_sensors(self):
        props = self._make_camera_props(
            self.WRIST_CAMERA_WIDTH, self.WRIST_CAMERA_HEIGHT, self.WRIST_CAMERA_FOV)
        self._wrist_camera_width = props.width
        self._wrist_camera_height = props.height
        self._wrist_camera_local_pos = torch.tensor(
            self.WRIST_CAMERA_POS, device=self.device, dtype=torch.float)
        self._wrist_camera_local_quat = torch.tensor(
            self.WRIST_CAMERA_QUAT, device=self.device, dtype=torch.float)
        self._wrist_camera_marker_geom = gymutil.WireframeSphereGeometry(
            0.025, 8, 8, None, color=(0.0, 1.0, 0.0))

        # The camera sits between the two finger pads, so it is driven explicitly
        # each frame from the midpoint of the two pad links (see
        # _update_wrist_camera_locations) rather than attached to one body.
        left_link, right_link = self.WRIST_CAMERA_BETWEEN_LINKS
        self._wrist_camera_handles = []
        self._wrist_between_body_handles = []   # [(left_handle, right_handle), ...]
        self._wrist_orient_body_handles = []
        for env_id in range(self.num_envs):
            cam = self.gym.create_camera_sensor(self.envs[env_id], props)
            left = self._find_camera_body_handle(env_id, left_link, "wrist (left pad)")
            right = self._find_camera_body_handle(env_id, right_link, "wrist (right pad)")
            orient = self._find_camera_body_handle(
                env_id, self.WRIST_CAMERA_ORIENT_LINK, "wrist (palm)")
            self._wrist_camera_handles.append(cam)
            self._wrist_between_body_handles.append((left, right))
            self._wrist_orient_body_handles.append(orient)

    def _create_head_camera_sensors(self):
        props = self._make_camera_props(
            self.HEAD_CAMERA_WIDTH, self.HEAD_CAMERA_HEIGHT, self.HEAD_CAMERA_FOV)
        self._head_camera_width = props.width
        self._head_camera_height = props.height
        self._head_camera_local_pos = torch.tensor(
            self.HEAD_CAMERA_POS, device=self.device, dtype=torch.float)
        self._head_camera_local_quat = torch.tensor(
            self.HEAD_CAMERA_QUAT, device=self.device, dtype=torch.float)
        self._head_camera_marker_geom = gymutil.WireframeSphereGeometry(
            0.035, 8, 8, None, color=(0.0, 1.0, 0.0))

        self._head_camera_handles = []
        self._head_camera_body_handles = []
        for env_id in range(self.num_envs):
            cam = self.gym.create_camera_sensor(self.envs[env_id], props)
            body = self._find_camera_body_handle(env_id, self.HEAD_CAMERA_LINK, "head")
            # attach_camera_to_body silently fails for non-zero envs when the
            # target is the base link (rigid-body index 0), so the head camera
            # is positioned explicitly each frame via set_camera_location (see
            # _update_head_camera_locations) instead of FOLLOW_TRANSFORM.
            self._head_camera_handles.append(cam)
            self._head_camera_body_handles.append(body)

    def _link_camera_pose(self, body_handle, local_pos, local_quat, env_id):
        """Camera world position + forward axis from a single parent link's pose."""
        body_state = self.rigid_body_state[env_id, body_handle]
        body_pos = body_state[:3]
        body_quat = body_state[3:7]
        cam_pos = body_pos + quat_apply(body_quat.unsqueeze(0), local_pos.unsqueeze(0))[0]
        forward_local = quat_apply(
            local_quat.unsqueeze(0),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0))[0]
        forward_world = quat_apply(body_quat.unsqueeze(0), forward_local.unsqueeze(0))[0]
        return cam_pos, forward_world

    def _wrist_camera_pose(self, env_id):
        """Camera pose at the midpoint of the two finger pads (between the
        fingers), aimed along the palm's grasp-approach axis."""
        left, right = self._wrist_between_body_handles[env_id]
        mid = 0.5 * (self.rigid_body_state[env_id, left, :3]
                     + self.rigid_body_state[env_id, right, :3])
        orient_quat = self.rigid_body_state[env_id, self._wrist_orient_body_handles[env_id], 3:7]
        cam_pos = mid + quat_apply(orient_quat.unsqueeze(0), self._wrist_camera_local_pos.unsqueeze(0))[0]
        forward_local = quat_apply(
            self._wrist_camera_local_quat.unsqueeze(0),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0))[0]
        forward_world = quat_apply(orient_quat.unsqueeze(0), forward_local.unsqueeze(0))[0]
        return cam_pos, forward_world

    def _aim_camera(self, cam_handle, env_id, cam_pos, forward_world):
        target = cam_pos + forward_world
        self.gym.set_camera_location(
            cam_handle, self.envs[env_id],
            gymapi.Vec3(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()),
            gymapi.Vec3(target[0].item(), target[1].item(), target[2].item()))

    def _update_wrist_camera_locations(self):
        for env_id in range(self.num_envs):
            cam_pos, forward = self._wrist_camera_pose(env_id)
            self._aim_camera(self._wrist_camera_handles[env_id], env_id, cam_pos, forward)

    def _update_head_camera_locations(self):
        for env_id in range(self.num_envs):
            cam_pos, forward = self._link_camera_pose(
                self._head_camera_body_handles[env_id],
                self._head_camera_local_pos, self._head_camera_local_quat, env_id)
            self._aim_camera(self._head_camera_handles[env_id], env_id, cam_pos, forward)

    def _read_camera_image(self, env_id, cam, image_type, height, width):
        img = np.asarray(self.gym.get_camera_image(self.sim, self.envs[env_id], cam, image_type))
        if image_type == gymapi.IMAGE_COLOR:
            if img.ndim == 1:
                img = img.reshape(height, width, 4)
            elif img.ndim == 2:
                img = img.reshape(img.shape[0], img.shape[1] // 4, 4)
            return img[:, :, :3].copy()          # drop alpha -> RGB
        # IMAGE_DEPTH: float32 (height, width), values are -Z distance (<= 0)
        if img.ndim == 1:
            img = img.reshape(height, width)
        return img.copy()

    def _read_all(self, handles, image_type, height, width):
        return np.stack(
            [self._read_camera_image(e, h, image_type, height, width)
             for e, h in enumerate(handles)], axis=0)

    def capture_cameras(self):
        """Render the camera sensors once and return their images.

        Returns a dict with any of: 'wrist_rgb', 'head_rgb', 'head_depth'
        (each shaped (num_envs, H, W, ...)). The head camera produces both an
        RGB image and a depth map from the same sensor.
        """
        out = {}
        if not self.enable_cameras:
            return out
        if self.record_wrist_camera:
            self._update_wrist_camera_locations()
        if self.record_head_camera:
            self._update_head_camera_locations()

        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        if self.record_wrist_camera:
            out['wrist_rgb'] = self._read_all(
                self._wrist_camera_handles, gymapi.IMAGE_COLOR,
                self._wrist_camera_height, self._wrist_camera_width)
        if self.record_head_camera:
            out['head_rgb'] = self._read_all(
                self._head_camera_handles, gymapi.IMAGE_COLOR,
                self._head_camera_height, self._head_camera_width)
            out['head_depth'] = self._read_all(
                self._head_camera_handles, gymapi.IMAGE_DEPTH,
                self._head_camera_height, self._head_camera_width)
        return out

    def _visualize_cameras(self):
        try:
            imgs = self.capture_cameras()
        except Exception as exc:
            print("camera read error:", exc)
            return
        if not hasattr(self, "_cam_windows"):
            self._cam_windows = set()

        def _show(name, frame):
            if name not in self._cam_windows:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                self._cam_windows.add(name)
            cv2.imshow(name, frame)

        if 'wrist_rgb' in imgs:
            _show("Wrist RGB", imgs['wrist_rgb'][0][:, :, ::-1])   # RGB -> BGR
        if 'head_rgb' in imgs:
            _show("Head RGB", imgs['head_rgb'][0][:, :, ::-1])
        if 'head_depth' in imgs:
            depth = -imgs['head_depth'][0]                         # to positive distance
            depth[~np.isfinite(depth)] = self.HEAD_DEPTH_FAR
            norm = np.clip(
                (depth - self.HEAD_DEPTH_NEAR) / (self.HEAD_DEPTH_FAR - self.HEAD_DEPTH_NEAR),
                0.0, 1.0)
            depth_u8 = (norm * 255).astype(np.uint8)
            _show("Head Depth", cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET))

    # ─────────────────────────────────────────────────────────────────────
    # Camera pose debug markers (drawn in the viewer)
    # ─────────────────────────────────────────────────────────────────────
    def _draw_marker_at(self, env_id, cam_pos, forward_world, marker_geom, length=0.25):
        # Green wireframe sphere at the camera position.
        pose = gymapi.Transform(
            gymapi.Vec3(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()), r=None)
        gymutil.draw_lines(marker_geom, self.gym, self.viewer, self.envs[env_id], pose)
        # Red line along the optical axis (camera forward), so orientation is visible.
        end = cam_pos + forward_world * length
        verts = np.array([[cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item(),
                           end[0].item(), end[1].item(), end[2].item()]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self.gym.add_lines(self.viewer, self.envs[env_id], 1, verts, colors)

    def _draw_camera_markers(self, env_id=0):
        """Draw a collision-free wireframe sphere + forward axis at each camera."""
        if self.viewer is None or not self.show_camera_markers:
            return
        if self.record_wrist_camera:
            cam_pos, forward = self._wrist_camera_pose(env_id)
            self._draw_marker_at(env_id, cam_pos, forward, self._wrist_camera_marker_geom)
        if self.record_head_camera:
            cam_pos, forward = self._link_camera_pose(
                self._head_camera_body_handles[env_id],
                self._head_camera_local_pos, self._head_camera_local_quat, env_id)
            self._draw_marker_at(env_id, cam_pos, forward, self._head_camera_marker_geom)

    # ─────────────────────────────────────────────────────────────────────
    # Buffers
    # ─────────────────────────────────────────────────────────────────────
    def _init_buffers(self):
        self.action_scale = torch.tensor(self.ACTION_SCALE, device=self.device)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot_dog")

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13)
        self.root_states = self._root_states[:, 0, :]
        self.box_root_state = self._root_states[:, 1, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_pos_wo_gripper = self.dof_pos[:, :-self.NUM_GRIPPER_DOF]
        self.dof_vel_wo_gripper = self.dof_vel[:, :-self.NUM_GRIPPER_DOF]

        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, :3]

        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.contact_forces = self._contact_forces[:, :-1, :]

        self._rigid_body_state_full = gymtorch.wrap_tensor(rigid_body_state_tensor).view(
            self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_state = self._rigid_body_state_full[:, :-1, :]

        self.jacobian_whole = gymtorch.wrap_tensor(jacobian_tensor)
        self.ee_pos = self.rigid_body_state[:, self.gripper_idx, :3]
        self.ee_orn = self.rigid_body_state[:, self.gripper_idx, 3:7]
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6,
                                             -(6 + self.NUM_GRIPPER_DOF):-self.NUM_GRIPPER_DOF]

        self.obs_history_buf = torch.zeros(self.num_envs, self.HISTORY_LEN, self.NUM_PROPRIO,
                                            device=self.device)
        self.action_history_buf = torch.zeros(self.num_envs, self.ACTION_DELAY + 2, self.NUM_ACTIONS,
                                                device=self.device)

        self.traj_timesteps = torch_rand_float(
            self.GOAL_TRAJ_TIME[0], self.GOAL_TRAJ_TIME[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(
            self.GOAL_HOLD_TIME[0], self.GOAL_HOLD_TIME[1], (self.num_envs, 1), device=self.device).squeeze(1) / self.dt
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_delta_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.curr_ee_goal_cart_world = torch.zeros(self.num_envs, 3, device=self.device)

        self.init_start_ee_sphere = torch.tensor(self.GOAL_INIT_POS_START, device=self.device).unsqueeze(0)
        self.init_end_ee_sphere = torch.tensor(self.GOAL_INIT_POS_END, device=self.device).unsqueeze(0)
        self.collision_lower_limits = torch.tensor(self.GOAL_COLLISION_LOWER, device=self.device)
        self.collision_upper_limits = torch.tensor(self.GOAL_COLLISION_UPPER, device=self.device)
        self.collision_check_t = torch.linspace(0, 1, self.GOAL_NUM_COLLISION_SAMPLES, device=self.device)[None, None, :]
        self.ee_goal_center_offset = torch.tensor(self.GOAL_SPHERE_CENTER_OFFSET, device=self.device).repeat(self.num_envs, 1)

        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat(self.num_envs, 1)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.p_gains = torch.zeros(self.NUM_TORQUES, device=self.device)
        self.d_gains = torch.zeros(self.NUM_TORQUES, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.NUM_ACTIONS, device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.NUM_GRIPPER_DOF, device=self.device)

        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.commands_scale = torch.tensor(
            [self.OBS_SCALES_LIN_VEL, self.OBS_SCALES_LIN_VEL, self.OBS_SCALES_ANG_VEL], device=self.device)

        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)

        self.base_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_yaw_quat[:, 3] = 1.0

        self.default_dof_pos = torch.zeros(self.num_dofs, device=self.device)
        for i in range(self.num_dofs):
            self.default_dof_pos[i] = self.DEFAULT_JOINT_ANGLES[self.dof_names[i]]
        self.default_dof_pos_wo_gripper = self.default_dof_pos[:-self.NUM_GRIPPER_DOF]

        for i in range(self.NUM_TORQUES):
            name = self.dof_names[i]
            for key in self.STIFFNESS.keys():
                if key in name:
                    self.p_gains[i] = self.STIFFNESS[key]
                    self.d_gains[i] = self.DAMPING[key]

        torque_limits_np = self.gym.get_asset_dof_properties(self.dog_asset)['effort']
        self.torque_limits = torch.tensor(torque_limits_np, dtype=torch.float, device=self.device)

        self.global_steps = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_episode_length = int(np.ceil(self.EPISODE_LENGTH_S / self.dt))

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.8, 1.2, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_vel[env_ids] = 0.
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        self.root_states[env_ids, :3] = to_torch(self.INIT_BASE_POS, device=self.device)
        self.root_states[env_ids, :2] += self.env_origin_xy(env_ids)
        self.root_states[env_ids, :2] += torch_rand_float(
            -self.ORIGIN_PERTURB_RANGE, self.ORIGIN_PERTURB_RANGE, (len(env_ids), 2), device=self.device)
        rand_yaw = self.RAND_YAW_RANGE * torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(0 * rand_yaw, 0 * rand_yaw, rand_yaw)
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -self.INIT_VEL_PERTURB_RANGE, self.INIT_VEL_PERTURB_RANGE, (len(env_ids), 6), device=self.device)

        base_quat = self.root_states[env_ids, 3:7]
        base_pos = self.root_states[env_ids, :3]
        offset_vec = torch.tensor(self.BOX_OFFSET, device=self.device)
        local_offset = offset_vec.unsqueeze(0).expand(len(env_ids), 3)
        self.box_root_state[env_ids, :3] = base_pos + quat_apply_yaw(base_quat, local_offset)
        self.box_root_state[env_ids, 3:7] = base_quat
        self.box_root_state[env_ids, 7:13] = 0.0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._resample_commands(env_ids)
        self._resample_ee_goal(env_ids, is_init=True)

        self.episode_length_buf[env_ids] = 0
        self.obs_history_buf[env_ids] = 0.
        self.action_history_buf[env_ids] = 0.
        self.goal_timer[env_ids] = 0.

    def env_origin_xy(self, env_ids):
        num_per_row = max(1, int(np.sqrt(self.num_envs)))
        spacing = 3.0
        out = torch.zeros(len(env_ids), 2, device=self.device)
        for k, e in enumerate(env_ids.tolist()):
            row, col = divmod(e, num_per_row)
            out[k, 0] = row * spacing
            out[k, 1] = col * spacing
        return out

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.step(torch.zeros(self.num_envs, self.NUM_ACTIONS, device=self.device))

    # ─────────────────────────────────────────────────────────────────────
    # Control helpers (ported from manip_loco.py)
    # ─────────────────────────────────────────────────────────────────────
    def _reindex_feet(self, vec):
        return vec[:, [1, 0, 3, 2]]

    def _reindex_all(self, vec):
        return torch.hstack((vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]], vec[:, 12:]))

    def _get_body_orientation(self):
        r, p, _ = euler_from_quat(self.base_quat)
        return torch.stack([r, p], dim=-1)

    def _control_ik(self, dpose):
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        A = torch.bmm(self.ee_j_eef, j_eef_T) + lmbda[None, ...]
        u = torch.bmm(j_eef_T, torch.linalg.solve(A, dpose))
        return u.squeeze(-1)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.action_scale
        default_torques = self.p_gains * (
            actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        default_torques[:, -6:] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_gripper_position_targets(self):
        return torch.zeros(
            self.num_envs, self.NUM_GRIPPER_DOF,
            dtype=self.dof_pos.dtype, device=self.device)

    def _get_ik_orn_target(self):
        # Desired EE orientation fed to the arm IK. Default keeps the original
        # behavior (gripper aligned to the base). Subclasses may override to
        # track a manual orientation goal.
        return self.root_states[:, 3:7]

    def _get_ee_goal_spherical_center(self):
        center = torch.cat([self.root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        return center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)

    def _get_walking_cmd_mask(self):
        m0 = torch.abs(self.commands[:, 0]) > self.LIN_VEL_X_CLIP
        m1 = torch.abs(self.commands[:, 1]) > self.LIN_VEL_X_CLIP
        m2 = torch.abs(self.commands[:, 2]) > self.ANG_VEL_YAW_CLIP
        return m0 | m1 | m2

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(
            0, self.LIN_VEL_X_RANGE[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0
        self.commands[env_ids, 2] = torch_rand_float(
            self.ANG_VEL_YAW_RANGE[0], self.ANG_VEL_YAW_RANGE[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :] *= (
            torch.logical_or(torch.abs(self.commands[env_ids, 0]) > self.LIN_VEL_X_CLIP,
                              torch.abs(self.commands[env_ids, 2]) > self.ANG_VEL_YAW_CLIP)).unsqueeze(1)

    def _collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(
            self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ..., None],
            self.collision_check_t).squeeze(-1)
        ee_target_cart = sphere2cart(
            torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)
        ).reshape(self.GOAL_NUM_COLLISION_SAMPLES, -1, 3)
        collision_mask = torch.any(
            torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1),
                               torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.GOAL_UNDERGROUND_LIMIT, dim=0)
        return collision_mask | underground_mask

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) == 0:
            return
        init_env_ids = env_ids.clone()
        if is_init:
            self.ee_goal_orn_delta_rpy[env_ids, :] = 0
            self.ee_start_sphere[env_ids] = self.init_start_ee_sphere
            self.ee_goal_sphere[env_ids] = self.init_end_ee_sphere
        else:
            r = torch_rand_float(self.GOAL_DELTA_ORN_R[0], self.GOAL_DELTA_ORN_R[1], (len(env_ids), 1), device=self.device)
            p = torch_rand_float(self.GOAL_DELTA_ORN_P[0], self.GOAL_DELTA_ORN_P[1], (len(env_ids), 1), device=self.device)
            y = torch_rand_float(self.GOAL_DELTA_ORN_Y[0], self.GOAL_DELTA_ORN_Y[1], (len(env_ids), 1), device=self.device)
            self.ee_goal_orn_delta_rpy[env_ids, :] = torch.cat([r, p, y], dim=-1)
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for _ in range(10):
                self.ee_goal_sphere[env_ids, 0] = torch_rand_float(
                    self.GOAL_POS_L[0], self.GOAL_POS_L[1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.ee_goal_sphere[env_ids, 1] = torch_rand_float(
                    self.GOAL_POS_P[0], self.GOAL_POS_P[1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.ee_goal_sphere[env_ids, 2] = torch_rand_float(
                    self.GOAL_POS_Y[0], self.GOAL_POS_Y[1], (len(env_ids), 1), device=self.device).squeeze(1)
                collision_mask = self._collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
        self.goal_timer[init_env_ids] = 0.0

    def _update_curr_ee_goal(self):
        t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.curr_ee_goal_cart[:] = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)
        self.curr_ee_goal_cart_world = self._get_ee_goal_spherical_center() + ee_goal_cart_yaw_global

        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + self.GOAL_ARM_INDUCED_PITCH
        self.ee_goal_orn_quat = quat_from_euler_xyz(
            self.ee_goal_orn_delta_rpy[:, 0] + np.pi / 2,
            default_pitch + self.ee_goal_orn_delta_rpy[:, 1],
            self.ee_goal_orn_delta_rpy[:, 2] + default_yaw)

        self.goal_timer += 1
        resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)

    def _step_contact_targets(self):
        frequencies = self.GAIT_FREQUENCY
        phases, offsets, bounds, durations = 0.5, 0.0, 0.0, 0.5
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
        self.gait_indices[~self._get_walking_cmd_mask()] = 0

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                         self.gait_indices + offsets,
                         self.gait_indices + bounds,
                         self.gait_indices + phases]
        for idxs in foot_indices:
            stance = torch.remainder(idxs, 1) < durations
            swing = torch.remainder(idxs, 1) > durations
            idxs[stance] = torch.remainder(idxs[stance], 1) * (0.5 / durations)
            idxs[swing] = 0.5 + (torch.remainder(idxs[swing], 1) - durations) * (0.5 / (1 - durations))

        for k in range(4):
            self.clock_inputs[:, k] = torch.sin(2 * np.pi * foot_indices[k])

    def _post_physics_step_callback(self):
        command_env_ids = (self.episode_length_buf % int(self.COMMAND_RESAMPLING_TIME / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(command_env_ids)
        self._step_contact_targets()

    # ─────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────
    def compute_observations(self):
        arm_base_offset = torch.tensor([0.3, 0., 0.09], device=self.device).repeat(self.num_envs, 1)
        arm_base_pos = self.base_pos + quat_apply(self.base_yaw_quat, arm_base_offset)
        ee_goal_local_cart = quat_rotate_inverse(self.base_quat, self.curr_ee_goal_cart_world - arm_base_pos)

        obs_buf = torch.cat((
            self._get_body_orientation(),
            self.base_ang_vel * self.OBS_SCALES_ANG_VEL,
            self._reindex_all((self.dof_pos - self.default_dof_pos) * self.OBS_SCALES_DOF_POS)[:, :-self.NUM_GRIPPER_DOF],
            self._reindex_all(self.dof_vel * self.OBS_SCALES_DOF_VEL)[:, :-self.NUM_GRIPPER_DOF],
            self._reindex_all(self.action_history_buf[:, -1])[:, :12],
            self._reindex_feet(self.foot_contacts_from_sensor),
            self.commands[:, :3] * self.commands_scale,
            ee_goal_local_cart,
            0 * self.curr_ee_goal_sphere,
        ), dim=-1)

        if self.OBSERVE_GAIT_COMMANDS:
            obs_buf = torch.cat((obs_buf, self.gait_indices.unsqueeze(1), self.clock_inputs), dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.HISTORY_LEN, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
        )

        policy_obs = torch.cat([obs_buf, self.obs_history_buf.reshape(self.num_envs, -1)], dim=-1)
        self.policy_obs = torch.clip(policy_obs, -self.CLIP_OBSERVATIONS, self.CLIP_OBSERVATIONS)

    def check_termination(self):
        r, p, _ = euler_from_quat(self.base_quat)
        z = self.root_states[:, 2]
        return (torch.abs(r) > self.R_TERM) | (torch.abs(p) > self.P_TERM) | (z < self.Z_TERM) | \
               (self.episode_length_buf > self.max_episode_length)

    # ─────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────
    def step(self, actions):
        actions = actions.clone()
        actions[:, 12:] = 0.
        actions = self._reindex_all(actions)
        actions = torch.clip(actions, -self.CLIP_ACTIONS, self.CLIP_ACTIONS).to(self.device)

        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
        actions = self.action_history_buf[:, -1]
        self.actions = actions.clone()

        dpos = self.curr_ee_goal_cart_world - self.ee_pos
        drot = orientation_error(self._get_ik_orn_target(), self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], -1).unsqueeze(-1)
        arm_pos_targets = self.dof_pos[:, -(6 + self.NUM_GRIPPER_DOF):-self.NUM_GRIPPER_DOF] + self._control_ik(dpose)
        all_pos_targets = torch.zeros_like(self.dof_pos)
        all_pos_targets[:, -(6 + self.NUM_GRIPPER_DOF):-self.NUM_GRIPPER_DOF] = arm_pos_targets
        all_pos_targets[:, -self.NUM_GRIPPER_DOF:] = self._get_gripper_position_targets()

        for _ in range(self.decimation):
            self.torques = self._compute_torques(self.actions)
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

    def post_physics_step(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.episode_length_buf += 1

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        base_yaw = euler_from_quat(self.base_quat)[2]
        self.base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(base_yaw), torch.zeros_like(base_yaw), base_yaw)

        # create_asset_force_sensor doesn't propagate to actors in this
        # isaacgym_tactile build (get_sim_force_sensor_count stays 0 even
        # right after create_actor), so contact is derived from net contact
        # forces instead, which is what manip_loco.py/legged_robot.py use
        # for contact detection elsewhere in training.
        self.foot_contacts_from_sensor = self.contact_forces[:, self.feet_indices, :].norm(dim=-1) > 1.5

        self._post_physics_step_callback()
        self._update_curr_ee_goal()

        reset_mask = self.check_termination()
        env_ids = reset_mask.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()
        self.tactile_sensor.reset_state_refresh_flag()

    # ─────────────────────────────────────────────────────────────────────
    # Tactile visualisation (same style as teleop_umi_ee.py)
    # ─────────────────────────────────────────────────────────────────────
    def get_tactile_observations(self, normalize=True):
        raw = self.tactile_sensor.get_tactile_observations()
        if not normalize:
            return raw
        result = []
        for obs in raw:
            obs = obs.clamp(min=0.0)
            max_val = obs.max(dim=1, keepdim=True)[0]
            mask = max_val < 0.0002
            normed = torch.where(mask, obs / 0.0008, obs / max_val.clamp(min=1e-9))
            result.append(normed)
        return result

    def _visualize_tactile(self):
        sensor_names = ["Left Tactile", "Right Tactile"]
        try:
            obs = self.get_tactile_observations(normalize=True)
        except Exception as exc:
            print("tactile read error:", exc)
            return

        if not hasattr(self, "_tac_windows"):
            self._tac_windows = set()

        for idx, name in enumerate(sensor_names):
            if idx >= len(obs):
                continue
            forces = obs[idx][0, :, 0].cpu().numpy()
            grid = forces.reshape(self.TACTILE_NUM_ROWS, self.TACTILE_NUM_COLUMNS)
            u8 = (np.clip(grid, 0.0, 1.0) * 255).astype(np.uint8)
            big = cv2.resize(u8, (self.TACTILE_NUM_COLUMNS * 20, self.TACTILE_NUM_ROWS * 20),
                              interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.applyColorMap(big, cv2.COLORMAP_VIRIDIS)
            if name not in self._tac_windows:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                self._tac_windows.add(name)
            cv2.imshow(name, heatmap)

    # ─────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────
    def render(self):
        if self.viewer is None:
            return
        if self.gym.query_viewer_has_closed(self.viewer):
            raise SystemExit("Viewer closed")
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "quit" and evt.value > 0:
                raise SystemExit("Quit requested")
        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        # Debug overlay must sit between step_graphics() and draw_viewer().
        if self.show_camera_markers and self.enable_cameras:
            self.gym.clear_lines(self.viewer)
            self._draw_camera_markers(env_id=0)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main(env_cls=PlayUMITac):
    parser = argparse.ArgumentParser(description="Replay manip_loco policy with tactile display")
    parser.add_argument("--proj_name", default="go2d1_low")
    parser.add_argument("--exptid", default="0323")
    parser.add_argument("--checkpoint", default="45000")
    parser.add_argument("--jit_policy_path", default=None,
                         help="Override: path to a *_jit.pt produced by export_jit_policy.py")
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

    jit_path = args.jit_policy_path
    if jit_path is None:
        jit_path = str(_LOW_LEVEL_ROOT / "logs" / args.proj_name / args.exptid / "traced" /
                        f"{args.exptid}_{args.checkpoint}_jit.pt")
    if not os.path.exists(jit_path):
        raise FileNotFoundError(
            f"No jit policy at {jit_path}.\n"
            f"Run once first:\n"
            f"  python legged_gym/scripts/export_jit_policy.py --task manip_loco "
            f"--proj_name {args.proj_name} --exptid {args.exptid} --checkpoint {args.checkpoint} "
            f"--observe_gait_commands --headless")

    env = env_cls(
        jit_policy_path=jit_path,
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
            obs = env.policy_obs
            with torch.no_grad():
                actions = env.policy(torch.cat(
                    (obs[:, :env.NUM_PROPRIO], obs[:, env.NUM_PROPRIO:]), dim=1))
            env.step(actions)

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
