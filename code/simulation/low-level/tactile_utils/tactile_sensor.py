# Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

"""
Standalone TactileSensor implementation.
"""

# CRITICAL: Isaac Gym must be imported before PyTorch!
# However, this module is designed to be imported after isaac_gym_wrapper,
# so Isaac Gym should already be loaded. We don't import it here to avoid issues.

import itertools
import numpy as np
import torch
import trimesh

from scipy.spatial.transform import Rotation as R
from urdfpy import URDF
from typing import List, Dict

# Support both relative import (when used as package) and absolute import (when run directly)
try:
    from .torch_utils import quat_apply, quat_conjugate, tf_apply, tf_combine, tf_inverse
except ImportError:
    from torch_utils import quat_apply, quat_conjugate, tf_apply, tf_combine, tf_inverse


class TactileSensor:
    """
    Standalone tactile sensor for Isaac Gym simulations.
    
    This class computes tactile forces using SDF (Signed Distance Field) queries
    between an elastomer surface and an indenter object.
    """

    # Physical parameters
    _ELASTOMER_RIGID_SHAPE_COMPLIANCE = 30.0
    _TACTILE_KN = 1.0  # Normal stiffness
    _TACTILE_KT = 0.1  # Tangential stiffness
    _TACTILE_MU = 2.0  # Friction coefficient

    def __init__(
        self,
        gym,
        sim,
        envs,
        actor_handles,
        num_envs,
        device,
        tactile_configs,
        tactile_num_rows,
        tactile_num_columns,
        tactile_point_distance,
    ):
        """
        Initialize tactile sensor.

        Args:
            gym: Isaac Gym gym object (gymapi.acquire_gym())
            sim: Isaac Gym sim instance
            envs: List of environment pointers
            actor_handles: List of dicts mapping actor names to handles
            num_envs: Number of parallel environments
            device: PyTorch device (e.g., 'cuda:0' or 'cpu')
            tactile_configs: List of sensor configuration dicts
            tactile_num_rows: Number of tactile points in row direction
            tactile_num_columns: Number of tactile points in column direction
            tactile_point_distance: Distance between tactile points (in meters)
        """
        self._gym = gym
        self._sim = sim
        self._envs = envs
        self._actor_handles = actor_handles
        self.num_envs = num_envs
        self.device = device

        # Tactile sensor configuration
        self._configs = tactile_configs
        self._TACTILE_NUM_ROWS = tactile_num_rows
        self._TACTILE_NUM_COLUMNS = tactile_num_columns
        self._TACTILE_POINT_DISTANCE = tactile_point_distance
        self._TACTILE_NUM_DIVS = (tactile_num_rows, tactile_num_columns)

        # Generate tactile points and load mesh origins
        self._generate_tactile_points()
        self._load_indenter_mesh_origin()

        # Will be initialized in post_load
        self._elastomer_link_handle = []
        self._indenter_link_handle = []
        self._indenter_shape_global_ids = []
        self._sdf_view = None
        self._simulator_link_state = None
        self._rigid_body_state = None
        self._link_state_refreshed = False
        self._created = False

    def _generate_tactile_points(self):
        """Generate grid of tactile sensing points on elastomer surface.
        
        Generates tactile points for each sensor separately to handle cases where
        sensors have different coordinate frames (e.g., mirrored fingers).
        """
        self._tactile_points_pos_local = []
        self._tactile_points_orn_local = []
        self._tactile_normal_axis = []
        
        for config in self._configs:
            robot = URDF.load(config["elastomer_urdf_path"])
            elastomer_mesh = (
                robot.link_map[config["elastomer_link_name"]].visuals[0].geometry.mesh.meshes[0]
            )

            # Generate grid points on elastomer flat plane
            grid_points = []
            elastomer_dims = np.diff(elastomer_mesh.bounds, axis=0).squeeze()
            slim_axis = elastomer_dims.argmin()  # The thin dimension (normal direction)
            center = (elastomer_mesh.bounds[0] + elastomer_mesh.bounds[1]) / 2.0
            idx = 0
            for axis_i in range(3):
                if axis_i == slim_axis:
                    # On the slim axis, place a point far away so ray is pointing at the elastomer tip
                    grid_points.append([-1.0])
                else:
                    axis_grid_points = np.linspace(
                        center[axis_i]
                        - self._TACTILE_POINT_DISTANCE * (self._TACTILE_NUM_DIVS[idx] + 1) / 2.0,
                        center[axis_i]
                        + self._TACTILE_POINT_DISTANCE * (self._TACTILE_NUM_DIVS[idx] + 1) / 2.0,
                        self._TACTILE_NUM_DIVS[idx] + 2,
                    )
                    # Leave out the extreme corners
                    grid_points.append(axis_grid_points[1:-1])
                    idx += 1
            grid_points = itertools.product(grid_points[0], grid_points[1], grid_points[2])
            grid_points = np.array(list(grid_points))

            # Project ray towards the elastomer mesh
            mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(elastomer_mesh)
            ray_directions = np.array([0.0, 0.0, 0.0])
            ray_directions[slim_axis] = +1.0
            ray_directions = np.tile([ray_directions], (grid_points.shape[0], 1))
            _, index_ray, locations = mesh_data.intersects_id(
                grid_points, ray_directions, return_locations=True, multiple_hits=False
            )

            tactile_points_pos_local = locations[index_ray.argsort()]
            tactile_points_pos_local = torch.tensor(
                tactile_points_pos_local, dtype=torch.float32, device=self.device
            )
            tactile_points_orn_local = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]] * tactile_points_pos_local.shape[0],
                dtype=torch.float32,
                device=self.device,
            )

            tactile_points_pos_local = tactile_points_pos_local.expand(self.num_envs, -1, -1)
            tactile_points_orn_local = tactile_points_orn_local.expand(self.num_envs, -1, -1)

            self._tactile_points_pos_local.append(tactile_points_pos_local)
            self._tactile_points_orn_local.append(tactile_points_orn_local)
            
            normal_axis = torch.eye(3, dtype=torch.float32, device=self.device)[slim_axis]
            self._tactile_normal_axis.append(normal_axis)

    def _load_indenter_mesh_origin(self):
        """Load the mesh origin transforms for all indenter objects."""
        self._indenter_mesh_pos_local = []
        self._indenter_mesh_orn_local = []

        for config in self._configs:
            # Handle procedurally generated geometries (e.g., spheres) without URDF
            if config["indenter_urdf_path"] is None:
                # Use identity transform for procedural geometries
                indenter_mesh_pos_local = torch.zeros(3, dtype=torch.float32, device=self.device)
                indenter_mesh_orn_local = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
            else:
                # Load from URDF for mesh-based objects
                robot = URDF.load(config["indenter_urdf_path"])
                origin = robot.link_map[config["indenter_link_name"]].visuals[0].origin
                indenter_mesh_pos_local = origin[:3, 3]
                indenter_mesh_pos_local = torch.as_tensor(
                    indenter_mesh_pos_local, dtype=torch.float32, device=self.device
                )
                indenter_mesh_orn_local = R.from_matrix(origin[:3, :3]).as_quat()
                indenter_mesh_orn_local = torch.as_tensor(
                    indenter_mesh_orn_local, dtype=torch.float32, device=self.device
                )
            
            indenter_mesh_pos_local = indenter_mesh_pos_local.expand(self.num_envs, -1)
            indenter_mesh_orn_local = indenter_mesh_orn_local.expand(self.num_envs, -1)

            self._indenter_mesh_pos_local.append(indenter_mesh_pos_local)
            self._indenter_mesh_orn_local.append(indenter_mesh_orn_local)

    def post_load_before_prepare(self):
        """
        Initialize sensor after assets are loaded but BEFORE prepare_sim().
        This must be called before prepare_sim() to properly setup SDF tensors.
        """
        if self._created:
            return

        self._load_simulator_handles()
        self._set_elastomer_compliant_dynamics()
        
        # Acquire SDF view tensor BEFORE prepare_sim
        self._acquire_sdf_view_tensor()

        self._created = True
    
    def post_load_after_prepare(self, rigid_body_state_tensor):
        """
        Finalize initialization after prepare_sim().
        
        Args:
            rigid_body_state_tensor: The wrapped rigid body state tensor from Isaac Gym
        """
        self._rigid_body_state = rigid_body_state_tensor

    def _load_simulator_handles(self):
        """Load Isaac Gym handles for elastomer and indenter rigid bodies."""
        self._elastomer_link_handle = []
        self._indenter_link_handle = []
        self._indenter_shape_global_ids = []

        # Import gymapi here to get DOMAIN_ACTOR
        from isaacgym_tactile import gymapi

        for config in self._configs:
            elastomer_rigid_body_handle = self._gym.find_actor_rigid_body_handle(
                self._envs[0],
                self._actor_handles[0][config["elastomer_actor_name"]],
                config["elastomer_link_name"],
            )
            self._elastomer_link_handle.append(elastomer_rigid_body_handle)

            indenter_rigid_body_handle = self._gym.find_actor_rigid_body_handle(
                self._envs[0],
                self._actor_handles[0][config["indenter_actor_name"]],
                config["indenter_link_name"],
            )
            self._indenter_link_handle.append(indenter_rigid_body_handle)

            indenter_rigid_shape_global_ids = torch.zeros(
                self.num_envs, dtype=torch.int32, device=self.device
            )
            for idx, env_ptr in enumerate(self._envs):
                indenter_actor_handle = self._actor_handles[idx][config["indenter_actor_name"]]
                indenter_rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
                    env_ptr, indenter_actor_handle
                )
                indenter_rigid_body_shape_indices = self._gym.get_actor_rigid_body_shape_indices(
                    env_ptr, indenter_actor_handle
                )
                indenter_rigid_body_idx = self._gym.find_actor_rigid_body_index(
                    env_ptr,
                    indenter_actor_handle,
                    config["indenter_link_name"],
                    gymapi.DOMAIN_ACTOR,
                )
                indenter_rigid_shape_idx = indenter_rigid_body_shape_indices[
                    indenter_rigid_body_idx
                ].start
                indenter_rigid_shape_global_ids[idx] = indenter_rigid_shape_props[
                    indenter_rigid_shape_idx
                ].global_index
            self._indenter_shape_global_ids.append(indenter_rigid_shape_global_ids)

    def _set_elastomer_compliant_dynamics(self):
        """Set compliant dynamics for elastomer surfaces."""
        from isaacgym_tactile import gymapi
        
        for config in self._configs:
            for idx, env_ptr in enumerate(self._envs):
                elastomer_actor_handle = self._actor_handles[idx][
                    config["elastomer_actor_name"]
                ]

                elastomer_rigid_shape_props = self._gym.get_actor_rigid_shape_properties(
                    env_ptr, elastomer_actor_handle
                )
                elastomer_rigid_body_shape_indices = self._gym.get_actor_rigid_body_shape_indices(
                    env_ptr, elastomer_actor_handle
                )
                elastomer_rigid_body_idx = self._gym.find_actor_rigid_body_index(
                    env_ptr,
                    elastomer_actor_handle,
                    config["elastomer_link_name"],
                    gymapi.DOMAIN_ACTOR,
                )
                elastomer_rigid_shape_idx = elastomer_rigid_body_shape_indices[
                    elastomer_rigid_body_idx
                ].start

                elastomer_rigid_shape_props[
                    elastomer_rigid_shape_idx
                ].compliance = self._ELASTOMER_RIGID_SHAPE_COMPLIANCE

                self._gym.set_actor_rigid_shape_properties(
                    env_ptr, elastomer_actor_handle, elastomer_rigid_shape_props
                )

    def _acquire_sdf_view_tensor(self):
        """Acquire SDF view tensor for distance queries."""
        from isaacgym_tactile import gymtorch

        num_queries_per_env = self._TACTILE_NUM_ROWS * self._TACTILE_NUM_COLUMNS
        sdf_view = self._gym.acquire_sdf_view_tensor(self._sim, 1, num_queries_per_env)
        self._sdf_view = gymtorch.wrap_tensor(sdf_view)

    def get_tactile_observations(self):
        """
        Compute tactile observations for all sensors.

        Returns:
            List of tactile observations, one per sensor.
            Each observation has shape (num_envs, num_points, 1) containing
            the normal force for each tactile point.
        """
        tactile_obs = []
        for sensor_idx, (
            elastomer_link_handle,
            indenter_link_handle,
            indenter_mesh_orn_local,
            indenter_mesh_pos_local,
            indenter_shape_global_ids,
            tactile_points_pos_local,
            tactile_points_orn_local,
            tactile_normal_axis,
        ) in enumerate(zip(
            self._elastomer_link_handle,
            self._indenter_link_handle,
            self._indenter_mesh_orn_local,
            self._indenter_mesh_pos_local,
            self._indenter_shape_global_ids,
            self._tactile_points_pos_local,
            self._tactile_points_orn_local,
            self._tactile_normal_axis,
        )):
            tactile_obs_sensor = self._get_tactile_observation_from_sensor(
                elastomer_link_handle,
                indenter_link_handle,
                indenter_mesh_orn_local,
                indenter_mesh_pos_local,
                indenter_shape_global_ids,
                tactile_points_pos_local,
                tactile_points_orn_local,
                tactile_normal_axis,
            )
            tactile_obs.append(tactile_obs_sensor)
        return tactile_obs

    def _get_tactile_observation_from_sensor(
        self,
        elastomer_link_handle,
        indenter_link_handle,
        indenter_mesh_orn_local,
        indenter_mesh_pos_local,
        indenter_shape_global_ids,
        tactile_points_pos_local,
        tactile_points_orn_local,
        tactile_normal_axis,
    ):
        """
        Compute tactile observation for a single sensor.

        This method computes the tactile forces based on:
        1. SDF queries to find penetration depth
        2. Normal and tangential force calculation based on spring-damper model
        3. Friction model
        """
        from isaacgym_tactile import gymtorch

        # Create link state view if not exists
        if not hasattr(self, "_simulator_link_state") or self._simulator_link_state is None:
            self._simulator_link_state = self._rigid_body_state.view(self.num_envs, -1, 13)

        # Refresh rigid body state if needed
        if not self._link_state_refreshed:
            self._gym.refresh_rigid_body_state_tensor(self._sim)
            self._link_state_refreshed = True

        # Get elastomer link pose in world frame
        elastomer_pos = self._simulator_link_state[:, elastomer_link_handle, 0:3]
        elastomer_orn = self._simulator_link_state[:, elastomer_link_handle, 3:7]
        elastomer_pos = elastomer_pos.unsqueeze(1).expand(
            -1, tactile_points_pos_local.shape[1], -1
        )
        elastomer_orn = elastomer_orn.unsqueeze(1).expand(
            -1, tactile_points_orn_local.shape[1], -1
        )

        # Get tactile points pose in world frame
        tactile_points_orn, tactile_points_pos = tf_combine(
            elastomer_orn,
            elastomer_pos,
            tactile_points_orn_local,
            tactile_points_pos_local,
        )

        # Get indenter link pose in world frame
        indenter_pos = self._simulator_link_state[:, indenter_link_handle, 0:3]
        indenter_orn = self._simulator_link_state[:, indenter_link_handle, 3:7]

        # Get indenter mesh pose in world frame
        indenter_mesh_orn, indenter_mesh_pos = tf_combine(
            indenter_orn, indenter_pos, indenter_mesh_orn_local, indenter_mesh_pos_local
        )
        indenter_mesh_orn = indenter_mesh_orn.unsqueeze(1).expand(
            -1, tactile_points_orn_local.shape[1], -1
        )
        indenter_mesh_pos = indenter_mesh_pos.unsqueeze(1).expand(
            -1, tactile_points_pos_local.shape[1], -1
        )

        # Get tactile points in indenter mesh frame
        indenter_mesh_orn_inv, indenter_mesh_pos_inv = tf_inverse(
            indenter_mesh_orn, indenter_mesh_pos
        )
        tactile_points_pos_indenter_mesh = tf_apply(
            indenter_mesh_orn_inv, indenter_mesh_pos_inv, tactile_points_pos
        )

        # Get signed distance and normal in indenter mesh frame
        torch.cuda.synchronize()
        self._gym.refresh_sdf_view_tensor(
            self._sim,
            gymtorch.unwrap_tensor(indenter_shape_global_ids),
            gymtorch.unwrap_tensor(tactile_points_pos_indenter_mesh.unsqueeze(1)),
        )

        signed_distance = self._sdf_view[:, :, :, 3].squeeze(1)

        # Return zero tactile normal force if signed distances are all non-negative
        if (signed_distance >= 0.0).all():
            tactile_normal_force = tactile_points_pos.new_zeros((*tactile_points_pos.shape[:-1], 1))
            tactile_obs = tactile_normal_force
            return tactile_obs

        # Get penetration depth
        depth = (-1.0 * signed_distance).clamp(min=0.0)

        # Get normal in indenter mesh and world frame
        normal_indenter_mesh = self._sdf_view[:, :, :, :3].squeeze(1)
        normal = quat_apply(indenter_mesh_orn, normal_indenter_mesh)

        # Get elastomer link velocity in world frame
        elastomer_linvel = self._simulator_link_state[:, elastomer_link_handle, 7:10]
        elastomer_angvel = self._simulator_link_state[:, elastomer_link_handle, 10:13]
        elastomer_linvel = elastomer_linvel.unsqueeze(1).expand(
            -1, tactile_points_pos_local.shape[1], -1
        )
        elastomer_angvel = elastomer_angvel.unsqueeze(1).expand(
            -1, tactile_points_orn_local.shape[1], -1
        )

        # Get tactile points linear velocity in world frame
        tactile_points_linvel = (
            torch.linalg.cross(
                elastomer_angvel, quat_apply(elastomer_orn, tactile_points_pos_local)
            )
            + elastomer_linvel
        )

        # Get indenter link velocity in world frame
        indenter_linvel = self._simulator_link_state[:, indenter_link_handle, 7:10]
        indenter_angvel = self._simulator_link_state[:, indenter_link_handle, 10:13]
        indenter_linvel = indenter_linvel.unsqueeze(1).expand(
            -1, tactile_points_pos_local.shape[1], -1
        )
        indenter_angvel = indenter_angvel.unsqueeze(1).expand(
            -1, tactile_points_orn_local.shape[1], -1
        )

        # Get closest points linear velocity in world frame
        closest_points_pos_indenter_mesh = (
            tactile_points_pos_indenter_mesh + depth.unsqueeze(-1) * normal_indenter_mesh
        )
        closest_points_linvel = (
            torch.linalg.cross(
                indenter_angvel, quat_apply(indenter_mesh_orn, closest_points_pos_indenter_mesh)
            )
            + indenter_linvel
        )

        # Get relative tangential velocity in world frame
        relative_linvel = tactile_points_linvel - closest_points_linvel
        relative_vt = relative_linvel - normal * (normal * relative_linvel).sum(
            dim=-1, keepdim=True
        )

        # Get tactile force in world frame
        fn_norm = self._TACTILE_KN * depth
        fn = fn_norm.unsqueeze(-1) * normal

        relative_vt_norm = relative_vt.norm(dim=-1)
        ft_static_norm = self._TACTILE_KT * relative_vt_norm
        ft_dynamic_norm = self._TACTILE_MU * fn_norm
        ft_norm = ft_static_norm.minimum(ft_dynamic_norm)
        ft = (
            -1.0
            * ft_norm.unsqueeze(-1)
            * relative_vt
            / relative_vt_norm.clamp(min=1e-9).unsqueeze(-1)
        )

        tactile_force = fn + ft

        # Get tactile force in tactile points frame
        tactile_force_local = quat_apply(quat_conjugate(tactile_points_orn), tactile_force)

        # Get tactile normal force
        tactile_normal_force = (tactile_force_local @ tactile_normal_axis).unsqueeze(-1)

        tactile_obs = tactile_normal_force

        return tactile_obs

    def reset_state_refresh_flag(self):
        """Reset the state refresh flag. Should be called after each simulation step."""
        self._link_state_refreshed = False

