import time
import json
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from go2_gym_deploy.utils.armstring import *
from go2_gym_deploy.utils.rc_command import *
from go2_gym_deploy.utils.wireless_controller import *
import threading
from collections import deque


def _axis_angle_to_rot(axis, theta):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)


def _rpy_to_rot(roll, pitch, yaw):
    """
    URDF rpy convention:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def _make_T(R=None, p=None):
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R

    if p is not None:
        T[:3, 3] = p

    return T


def compute_ee_jacobian(q, return_ee_pose=False):
    """
    Compute the end-effector geometric Jacobian from the D1 URDF.

    Args:
        q: np.ndarray, shape (6,)
           Current joint angles in radians.
           Order:
           [Z_Joint1, Z_Joint2, Z_Joint3,
            Z_Joint4, Z_Joint5, Z_Joint6]

        return_ee_pose: bool
           If True, also return p_ee and R_ee.

    Returns:
        J: np.ndarray, shape (6, 6)
           J[:3, :] is the linear Jacobian.
           J[3:, :] is the angular Jacobian.
           Both are expressed in the base frame.

        If return_ee_pose=True:
           return J, p_ee, R_ee
    """

    q = np.asarray(q, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"q should have shape (6,), got {q.shape}")

    # joint origin xyz from URDF
    joint_xyz = [
        [0.0,      0.0,       0.0738],
        [0.0,     -0.0276,    0.0578],
        [0.0,     -0.0004,    0.2700],
        [0.05,     0.0275,    0.041325],
        [0.15468, -0.0258,    0.0001],
        [0.0777,   0.025822, -0.0010718],
    ]

    # joint axis from URDF, expressed in each joint frame
    joint_axis = [
        [0.0, 0.0, -1.0],   # Z_Joint1
        [0.0, 1.0,  0.0],   # Z_Joint2
        [0.0, 1.0,  0.0],   # Z_Joint3
        [1.0, 0.0,  0.0],   # Z_Joint4
        [0.0, 1.0,  0.0],   # Z_Joint5
        [1.0, 0.0,  0.0],   # Z_Joint6
    ]

    joint_xyz = [np.array(x, dtype=np.float64) for x in joint_xyz]
    joint_axis = [np.array(a, dtype=np.float64) for a in joint_axis]

    # fixed transform from Empty_Link6 to end_effector
    ee_xyz = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    ee_rpy = np.array([0.0, -1.57, 0.0], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)

    joint_pos_base = []
    joint_axis_base = []

    for i in range(6):
        T_joint = T @ _make_T(p=joint_xyz[i])
        p_i = T_joint[:3, 3].copy()
        z_i = T_joint[:3, :3] @ joint_axis[i]
        z_i = z_i / np.linalg.norm(z_i)

        joint_pos_base.append(p_i)
        joint_axis_base.append(z_i)

        R_joint = _axis_angle_to_rot(joint_axis[i], q[i])
        T = T_joint @ _make_T(R=R_joint)

    # Empty_Link6 -> end_effector fixed joint
    R_ee_fixed = _rpy_to_rot(*ee_rpy)
    T_ee = T @ _make_T(R=R_ee_fixed, p=ee_xyz)

    p_ee = T_ee[:3, 3].copy()
    R_ee = T_ee[:3, :3].copy()

    J = np.zeros((6, 6), dtype=np.float64)

    for i in range(6):
        z_i = joint_axis_base[i]
        p_i = joint_pos_base[i]

        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i

    if return_ee_pose:
        return J, p_ee, R_ee

    return J


def control_ik(J, dpose, damping=0.05):
    J = np.asarray(J, dtype=np.float64)
    dpose = np.asarray(dpose, dtype=np.float64)
    if dpose.ndim == 1:
        dpose = dpose.reshape(6, 1)
    J_T = J.T
    lmbda = np.eye(6) * (damping ** 2)
    A = J @ J_T + lmbda
    dq = J_T @ np.linalg.solve(A, dpose)
    return dq.squeeze(-1)

class ArmController:
    def __init__(self, netorkinterface='eno2'):
        # self.default_dof_pos = [0, -1.57, 1.57, 0, 0, 0, 0]

        self.default_dof_pos = [0.00523599,  0.97563908, -0.21118485, -0.06632251,  0.52359878, -0.00872665, 0]
        self.smoothing_ratio = 0.2
        self.dof_pos = np.zeros(6)
        self.last_dof_pos = np.zeros(6)
        self.dof_vel = np.zeros(6)
        self.dof_pos_history = deque([], maxlen=5)
        self.dt_history = deque([], maxlen=5)
        self.action = np.zeros(6)
        self.ee_pos = np.zeros(3)
        self.ee_rot = np.zeros(3)
        self.ee_target = np.zeros(6)
        self.delta_ee = np.zeros(6)

        self.a = np.array([0.1043, 0.0003, 0.0483])

        self.last_time = time.time()
        self.command = None
        ChannelFactoryInitialize(0, netorkinterface)
        print('Connection Successful')

        self.pub = ChannelPublisher("rt/arm_Command", ArmString_)
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/arm_Feedback", ArmString_)
        self.sub.Init(self.state_estimator, queueLen=10)


        self.sub2 = ChannelSubscriber("rt/arm_Feedback", ArmString_)
        self.sub2.Init(self.control, queueLen=10)

        self.rc_command_sub = ChannelSubscriber("rt/lf/lowstate", LowState_)
        self.rc_command_sub.Init(self.rc_command, queueLen=10)

        self.remote_controller = unitreeRemoteController()


    def reset(self):
        print('Reseting')
        while(1):
            msg = ArmString_(" ")
            data_ = {"seq":4,
                     "address": 1,
                     "funcode": 2,
                     "data":{"mode": 1,
                             "angle0": np.rad2deg(self.default_dof_pos[0]),
                             "angle1": np.rad2deg(self.default_dof_pos[1]),
                             "angle2": np.rad2deg(self.default_dof_pos[2]),
                             "angle3": np.rad2deg(self.default_dof_pos[3]),
                             "angle4": np.rad2deg(self.default_dof_pos[4]),
                             "angle5": np.rad2deg(self.default_dof_pos[5]),
                             "angle6": np.rad2deg(self.default_dof_pos[6]),
                             "delay_ms": 0}}
            msg.data_ = json.dumps(data_)
            # Publish message
            if self.pub.Write(msg, 0.5):
                pass
            else:
                print("Waitting for subscriber.")

            error = np.mean(np.abs(self.get_arm_dof_pos() - self.default_dof_pos[:-1]))
            if error <= 0.01:
                print('Successfully reset')
                break

    def ee_delta_control(self):
        joint_pos_curr = np.array(self.get_arm_dof_pos(), dtype=np.float64)
        J = compute_ee_jacobian(joint_pos_curr)

        self.action = self.ee_target - np.concatenate([self.get_ee_pos().copy(), self.get_ee_rot().copy()], axis=-1)
        # print(self.ee_target[:3], self.get_ee_pos(), self.action)
        ee_pos_delta = np.array(self.action.copy(), dtype=np.float64)

        joint_pos_delta = control_ik(J, ee_pos_delta)
        joint_pos_delta = np.hstack([self.get_arm_dof_pos() + joint_pos_delta, np.zeros(1, )])
        self.forward(joint_pos_delta)

    def forward(self, angle):
        for i in range(1):
            # Create a Userdata message
            msg = ArmString_(" ")
            data_ = {"seq":4,
                     "address": 1,
                     "funcode": 2,
                     "data":{"mode": 1,
                             "angle0": np.rad2deg(angle[0]),
                             "angle1": np.rad2deg(angle[1]),
                             "angle2": np.rad2deg(angle[2]),
                             "angle3": np.rad2deg(angle[3]),
                             "angle4": np.rad2deg(angle[4]),
                             "angle5": np.rad2deg(angle[5]),
                             "angle6": np.rad2deg(angle[6]),
                             "delay_ms": 0}}
            msg.data_ = json.dumps(data_)
            if self.pub.Write(msg, 0.5):
                pass
            else:
                print("Waitting for subscriber.")


    def get_arm_dof_pos(self):
        return self.dof_pos

    def get_arm_dof_vel(self):
        return self.dof_vel

    def get_ee_pos(self):
        return self.ee_pos

    def get_ee_rot(self):
        return self.ee_rot

    def enable_joint(self):
        msg = ArmString_(" ")
        msg.data_ = '{"seq":4,"address":1,"funcode":5,"data":{"mode": 1}}'
        if self.pub.Write(msg, 0.5):
            print("Successfully enable joint")
        else:
            print("Failed to send command")

    def disable_joint(self):
        msg = ArmString_(" ")
        msg.data_ = '{"seq":4,"address":1,"funcode":5,"data":{"mode": 0}}'
        if self.pub.Write(msg, 0.5):
            print("Successfully disable joint")
        else:
            print("Failed to send command")

    def process_command(self):
        if self.command == 'reset':
            self.command = None
            self.reset()
        elif self.command == 'enable':
            self.command = None
            self.enable_joint()
        elif self.command == 'disable':
            self.command = None
            self.disable_joint()

    def state_estimator(self, msg):
        # try:
        #     while True:
        now = time.time()
        # msg = self.sub.Read(timeout=0.5)
        json_msg = json.loads(msg.data_)
        data = json_msg["data"]
        if "angle0" in list(data.keys()):
            dof_state = np.asarray([data[f'angle{index}'] for index in range(0, 6)])
            dof_state = np.deg2rad(dof_state)
            self.dof_pos_history.append(dof_state - self.last_dof_pos)
            self.last_dof_pos = self.dof_pos.copy()
            self.dof_pos = dof_state

            dt = now - self.last_time
            self.dt_history.append(dt)

            dof_pos_history = np.array(self.dof_pos_history, dtype=np.float64).reshape(-1, 6)
            dt_history = np.array(self.dt_history, dtype=np.float64).clip(min=0.1).reshape(-1, 1)

            dof_vel = (dof_pos_history / dt_history).mean(axis=0)
            self.dof_vel = self.smoothing_ratio * dof_vel + (1 - self.smoothing_ratio) * self.dof_vel
        self.last_time = now
        _, ee_pos, _ = compute_ee_jacobian(self.dof_pos, return_ee_pose=True)
        self.ee_pos = ee_pos

        # except KeyboardInterrupt:
        #     pass

    def control(self, msg):
        if self.command is not None:
            self.process_command()
        if np.any(self.delta_ee):
            self.ee_target = np.concatenate([self.get_ee_pos().copy(), self.get_ee_rot().copy()], axis=-1) + self.delta_ee.copy()
            self.action = self.delta_ee.copy()
            self.delta_ee = np.zeros(3)
        if np.sum(np.abs(self.action)) >= 0.01:
            self.ee_delta_control()
        else:
            self.ee_target = np.concatenate([self.get_ee_pos().copy(), self.get_ee_rot().copy()],axis=-1)
        # self.ee_target = np.concatenate([self.get_ee_pos().copy(), self.get_ee_rot().copy()], axis=-1) + self.delta_ee.copy()


    def rc_command(self, msg):
        wireless_remote_data = msg.wireless_remote
        self.remote_controller.parse(wireless_remote_data)
        if self.remote_controller.Up == 1:
            # self.delta_ee = np.array([0.1, 0., 0., 0, 0, 0])
            self.a += np.array([0.03, 0., 0.])
        elif self.remote_controller.Down == 1:
            self.a -= np.array([0.03, 0., 0.])
            # self.delta_ee = np.array([-0.1, 0., 0., 0, 0, 0])
        elif self.remote_controller.Left == 1:
            self.a += np.array([0., 0.03, 0.])
            # self.delta_ee = np.array([0., 0.1, 0., 0, 0, 0])
        elif self.remote_controller.Right == 1:
            self.a -= np.array([0., 0.03, 0.])
            # self.delta_ee = np.array([0., -0.1, 0., 0, 0, 0])
        elif self.remote_controller.Y == 1:
            self.a += np.array([0., 0., 0.03])
            # self.delta_ee = np.array([0., 0., 0.1, 0, 0, 0])
        elif self.remote_controller.A == 1:
            self.a -= np.array([0., 0., 0.03])
            # self.delta_ee = np.array([0., 0., -0.1, 0, 0, 0])
        elif self.remote_controller.Select == 1:
            self.command = 'reset'

    def spin(self):
        # self.thread1 = threading.Thread(target=self.state_estimator, daemon=False)
        # self.thread1.start()

        self.thread = threading.Thread(target=self.control, daemon=False)
        self.thread.start()

    def close(self):
        self.pub.Close()
        self.sub.Close()

if __name__ == "__main__":
    controller = ArmController()
    # controller.disable_joint()
    # controller.enable_joint()
    # controller.command = 'enable'
    # controller.spin()
    # angle = [0, 0, 1.57, 0, 0, 0, 0]
    # controller.forward(angle)
    # action = [0.1, 0., 0, 0, 0, 0]
    # controller.ee_delta_control()
    # time.sleep(1)
    while(1):
        print(controller.get_ee_pos())
        time.sleep(1)
    # controller.action = np.array([-0.1, 0.1, 0.1, 0, 0, 0])
    # controller.command = 'reset'
    # print(controller.action)

    # controller.enable_joint()
    # controller.get_arm_dof_pos()
    # controller.forward(angle)
    # controller.disable_joint()
    # action = [0.1, 0., 0, 0, 0, 0]
    # controller.ee_delta_control(action)


