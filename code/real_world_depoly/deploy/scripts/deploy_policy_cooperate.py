import argparse
import lcm
from pathlib import Path

import torch

from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
from go2_gym_deploy.envs.lcm_agent import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go2_gym_deploy.utils.arm_controller import ArmController
from go2_gym_deploy.utils.command_profile import *
from go2_gym_deploy.scripts.cooperate_deploy_config import COOPERATE_CFG


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CHECKPOINT_DIR = SCRIPT_DIR.parents[2] / "pretrained_checkpoint"
ACTOR_BODY_CKPT = CHECKPOINT_DIR / "body_latest.jit"
ADAPTATION_MODULE_CKPT = CHECKPOINT_DIR / "adaptation_module_latest.jit"

LCM_URL = "udpm://239.255.76.67:7667?ttl=255"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="enx6c6e072d458f", help="network interface for Unitree SDK2 Python")
    parser.add_argument("--max-steps", type=int, default=10000000)
    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--max-yaw-vel", type=float, default=1.0)
    parser.add_argument("--experiment-name", default="example_experiment")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    return parser.parse_args()


def load_and_run_policy(checkpoint_dir, experiment_name, network_interface, max_steps, max_vel=1.0, max_yaw_vel=1.0):
    cfg = COOPERATE_CFG
    print(list(cfg.keys()))
    print('Config successfully loaded!')

    lc = lcm.LCM(LCM_URL)
    se = StateEstimator(lc)
    ac = ArmController(netorkinterface=network_interface)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, ac, command_profile)
    se.spin()
    # ac.spin()

    from go2_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)
    print('Agent successfully created!')

    policy = load_policy(checkpoint_dir)
    print('Policy successfully loaded!')

    # load runner
    root = REPO_ROOT / "logs"
    root.mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=str(root / experiment_name))
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    body_ckpt = checkpoint_dir / ACTOR_BODY_CKPT.name
    adaptation_ckpt = checkpoint_dir / ADAPTATION_MODULE_CKPT.name

    for ckpt in (body_ckpt, adaptation_ckpt):
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    print(f"Loading body checkpoint: {body_ckpt}")
    body = torch.jit.load(str(body_ckpt), map_location="cpu").to('cpu')

    print(f"Loading adaptation checkpoint: {adaptation_ckpt}")
    adaptation_module = torch.jit.load(str(adaptation_ckpt), map_location="cpu").to('cpu')

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy




if __name__ == '__main__':
    args = parse_args()
    load_and_run_policy(
        args.checkpoint_dir,
        experiment_name=args.experiment_name,
        network_interface=args.net,
        max_steps=args.max_steps,
        max_vel=args.max_vel,
        max_yaw_vel=args.max_yaw_vel,
    )
