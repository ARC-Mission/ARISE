#!/usr/bin/env python3
"""
ARC-M: Autonomous Recovery Training Script
Implements Eureka-style LLM-guided reward generation with Isaac Lab

Usage:
    python train_recovery_policy.py --num_envs 4096 --use_magistral
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add Isaac Lab to path
ISAAC_LAB_PATH = os.environ.get("ISAAC_LAB_PATH", os.path.expanduser("~/IsaacLab"))
sys.path.insert(0, ISAAC_LAB_PATH)

# Isaac Lab imports (after path setup)
from omni.isaac.lab.app import AppLauncher

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="ARC-M Recovery Policy Training")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum training iterations")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--use_magistral", action="store_true", help="Enable LLM-guided reward generation")
parser.add_argument("--config", type=str, default="config/isaac_lab_config.yaml", help="Config file path")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import everything else (after Isaac Sim is running)
import torch
import yaml
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Isaac Lab imports
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

# RL imports
try:
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.modules import ActorCritic
    RSL_RL_AVAILABLE = True
except ImportError:
    RSL_RL_AVAILABLE = False
    print("[WARNING] rsl_rl not available, using stable-baselines3")


class MagistralRewardGenerator:
    """
    Eureka-style LLM-guided reward function generation.
    Uses Mistral Magistral to generate and refine reward functions.
    """
    
    SYSTEM_PROMPT = """You are an expert reward engineer for reinforcement learning in robotics.
Your task is to design reward functions that train quadruped robots for recovery behaviors.

When generating rewards, consider:
1. Primary objectives (task completion)
2. Stability constraints (don't fall over)
3. Energy efficiency (minimize torque)
4. Safety (avoid self-collision, limit velocities)

Output format: Python code defining a reward function that takes observation dict and returns scalar reward.
Use PyTorch operations for GPU compatibility.
Include comments explaining each reward component."""

    def __init__(self, model: str = "magistral-medium-latest", use_api: bool = True):
        self.model = model
        self.use_api = use_api
        self.client = None
        
        if use_api:
            try:
                from mistralai import Mistral
                api_key = os.environ.get("MISTRAL_API_KEY")
                if not api_key:
                    raise ValueError("MISTRAL_API_KEY not set")
                self.client = Mistral(api_key=api_key)
            except ImportError:
                print("[WARNING] mistralai not installed, reward generation disabled")
                
    def generate_reward_function(
        self, 
        task_description: str,
        observation_space: Dict[str, Any],
        previous_results: Optional[Dict] = None
    ) -> str:
        """Generate a reward function based on task description and training feedback."""
        
        if not self.client:
            return self._default_reward_function()
            
        # Build prompt
        prompt = f"""Task Description:
{task_description}

Observation Space:
{yaml.dump(observation_space, default_flow_style=False)}
"""
        
        if previous_results:
            prompt += f"""
Previous Training Results:
- Mean reward: {previous_results.get('mean_reward', 'N/A')}
- Success rate: {previous_results.get('success_rate', 'N/A')}
- Average episode length: {previous_results.get('episode_length', 'N/A')}
- Issues observed: {previous_results.get('issues', 'None')}

Please analyze these results and improve the reward function."""

        prompt += """

Generate a Python reward function. Output ONLY the Python code wrapped in ```python``` tags.
The function should be named `compute_reward` and take (obs_dict, actions, reset_buf) as arguments."""

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            # Extract code from response
            content = response.choices[0].message.content
            code = self._extract_code(content)
            return code
            
        except Exception as e:
            print(f"[WARNING] Magistral generation failed: {e}")
            return self._default_reward_function()
    
    def _extract_code(self, content: str) -> str:
        """Extract Python code from markdown code blocks."""
        import re
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches[0] if matches else self._default_reward_function()
    
    def _default_reward_function(self) -> str:
        """Fallback reward function."""
        return '''
def compute_reward(obs_dict, actions, reset_buf):
    """Default recovery reward function."""
    import torch
    
    # Unpack observations
    base_lin_vel = obs_dict["base_lin_vel"]
    base_ang_vel = obs_dict["base_ang_vel"]
    projected_gravity = obs_dict["projected_gravity"]
    dof_pos = obs_dict["dof_pos"]
    dof_vel = obs_dict["dof_vel"]
    commands = obs_dict.get("commands", torch.zeros_like(base_lin_vel))
    
    # Velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    tracking_reward = torch.exp(-lin_vel_error / 0.25) + 0.5 * torch.exp(-ang_vel_error / 0.25)
    
    # Stability reward (penalize non-upright orientation)
    orientation_penalty = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    
    # Energy efficiency (penalize high torques/accelerations)
    torque_penalty = torch.sum(torch.square(actions), dim=1) * 0.0002
    
    # Action smoothness
    action_rate_penalty = torch.sum(torch.square(actions), dim=1) * 0.01
    
    # Combine rewards
    reward = (
        tracking_reward * 1.5
        - orientation_penalty * 0.5
        - torque_penalty
        - action_rate_penalty
    )
    
    return reward
'''


class RecoveryEnvConfig:
    """Configuration class for the recovery environment."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    @property
    def num_envs(self) -> int:
        return self.config['env']['num_envs']
    
    @property
    def observation_space(self) -> Dict:
        return self.config['observations']
    
    @property
    def training_config(self) -> Dict:
        return self.config['training']
    
    @property
    def magistral_config(self) -> Dict:
        return self.config.get('magistral', {})


def create_training_env(cfg: RecoveryEnvConfig, num_envs: int, headless: bool):
    """Create Isaac Lab training environment."""
    
    # For now, use a built-in locomotion task as base
    # In production, this would be a custom ARC-M environment
    task_name = "Isaac-Velocity-Flat-Anymal-D-v0"
    
    env = gym.make(
        task_name,
        num_envs=num_envs,
        device="cuda",
    )
    
    # Wrap for RSL-RL
    if RSL_RL_AVAILABLE:
        env = RslRlVecEnvWrapper(env)
    
    return env


def train_with_magistral(
    env,
    cfg: RecoveryEnvConfig,
    reward_generator: MagistralRewardGenerator,
    max_iterations: int,
    checkpoint_dir: str
):
    """Training loop with Eureka-style reward refinement."""
    
    magistral_cfg = cfg.magistral_config
    reward_iterations = magistral_cfg.get('reward_generation', {}).get('iterations', 5)
    analyze_interval = magistral_cfg.get('feedback', {}).get('analyze_interval', 500)
    
    task_description = magistral_cfg.get('task_description', 'Train a quadruped for recovery')
    
    # Initial reward generation
    print("\n[INFO] Generating initial reward function with Magistral...")
    reward_code = reward_generator.generate_reward_function(
        task_description=task_description,
        observation_space=cfg.observation_space
    )
    
    # Save generated reward
    reward_path = os.path.join(checkpoint_dir, "reward_function.py")
    with open(reward_path, 'w') as f:
        f.write(reward_code)
    print(f"[INFO] Reward function saved to: {reward_path}")
    
    # Training configuration
    train_cfg = cfg.training_config
    
    if RSL_RL_AVAILABLE:
        # Use RSL-RL PPO (optimized for Isaac Lab)
        from rsl_rl.algorithms import PPO
        
        # Create actor-critic network
        actor_critic = ActorCritic(
            num_obs=env.num_obs,
            num_actions=env.num_actions,
            actor_hidden_dims=train_cfg['policy']['hidden_dims'],
            critic_hidden_dims=train_cfg['value']['hidden_dims'],
            activation=train_cfg['policy']['activation'],
        ).to("cuda")
        
        # Create PPO algorithm
        ppo = PPO(
            actor_critic=actor_critic,
            num_learning_epochs=train_cfg['num_epochs'],
            num_mini_batches=train_cfg['num_minibatches'],
            clip_param=train_cfg['clip_param'],
            gamma=train_cfg['gamma'],
            lam=train_cfg['lam'],
            learning_rate=train_cfg['learning_rate'],
            entropy_coef=train_cfg['entropy_coef'],
        )
        
        # Create runner
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=checkpoint_dir,
            device="cuda"
        )
        
        # Training loop with reward refinement
        current_iteration = 0
        training_results = {}
        
        for reward_iter in range(reward_iterations):
            print(f"\n{'='*60}")
            print(f"Reward Iteration {reward_iter + 1}/{reward_iterations}")
            print(f"{'='*60}")
            
            # Train for analyze_interval iterations
            iterations_this_round = min(
                analyze_interval,
                max_iterations - current_iteration
            )
            
            if iterations_this_round <= 0:
                break
                
            # Run training
            runner.learn(num_learning_iterations=iterations_this_round)
            current_iteration += iterations_this_round
            
            # Collect metrics for Magistral feedback
            training_results = {
                'mean_reward': runner.alg.mean_reward,
                'success_rate': 0.0,  # Would need custom tracking
                'episode_length': runner.alg.mean_episode_length,
                'issues': []
            }
            
            # Check for issues
            if training_results['mean_reward'] < 0:
                training_results['issues'].append("Negative mean reward - policy may be unstable")
            if training_results['episode_length'] < 100:
                training_results['issues'].append("Short episodes - may be falling frequently")
            
            # Regenerate reward with feedback
            if reward_iter < reward_iterations - 1 and magistral_cfg.get('feedback', {}).get('enabled', True):
                print("\n[INFO] Analyzing results and refining reward function...")
                reward_code = reward_generator.generate_reward_function(
                    task_description=task_description,
                    observation_space=cfg.observation_space,
                    previous_results=training_results
                )
                
                # Save updated reward
                reward_path = os.path.join(checkpoint_dir, f"reward_function_iter{reward_iter+1}.py")
                with open(reward_path, 'w') as f:
                    f.write(reward_code)
                print(f"[INFO] Updated reward saved to: {reward_path}")
        
        # Save final model
        runner.save(os.path.join(checkpoint_dir, "final_model"))
        
    else:
        # Fallback to Stable-Baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_cfg['learning_rate'],
            n_steps=2048,
            batch_size=64,
            n_epochs=train_cfg['num_epochs'],
            gamma=train_cfg['gamma'],
            gae_lambda=train_cfg['lam'],
            clip_range=train_cfg['clip_param'],
            ent_coef=train_cfg['entropy_coef'],
            verbose=1,
            tensorboard_log=checkpoint_dir
        )
        
        model.learn(total_timesteps=max_iterations * env.num_envs)
        model.save(os.path.join(checkpoint_dir, "final_model"))


def main():
    """Main training entry point."""
    
    print("\n" + "="*60)
    print("ARC-M: Autonomous Recovery Training")
    print("="*60 + "\n")
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    config_path = project_dir / args_cli.config
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = project_dir / "logs" / f"arc_m_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {config_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Num envs: {args_cli.num_envs}")
    print(f"Use Magistral: {args_cli.use_magistral}")
    
    # Load configuration
    cfg = RecoveryEnvConfig(str(config_path))
    
    # Override num_envs from command line
    cfg.config['env']['num_envs'] = args_cli.num_envs
    
    # Create environment
    print("\n[INFO] Creating training environment...")
    env = create_training_env(cfg, args_cli.num_envs, args_cli.headless)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Setup reward generator
    reward_generator = None
    if args_cli.use_magistral:
        magistral_cfg = cfg.magistral_config
        reward_generator = MagistralRewardGenerator(
            model=magistral_cfg.get('model', 'magistral-medium-latest'),
            use_api=True
        )
    
    # Train
    print("\n[INFO] Starting training...")
    
    if reward_generator:
        train_with_magistral(
            env=env,
            cfg=cfg,
            reward_generator=reward_generator,
            max_iterations=args_cli.max_iterations,
            checkpoint_dir=str(checkpoint_dir)
        )
    else:
        # Simple training without LLM guidance
        if RSL_RL_AVAILABLE:
            from rsl_rl.runners import OnPolicyRunner
            runner = OnPolicyRunner(
                env=env,
                train_cfg=cfg.training_config,
                log_dir=str(checkpoint_dir),
                device="cuda"
            )
            runner.learn(num_learning_iterations=args_cli.max_iterations)
            runner.save(str(checkpoint_dir / "final_model"))
    
    print("\n[INFO] Training complete!")
    print(f"Model saved to: {checkpoint_dir}")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
