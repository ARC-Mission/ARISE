"""
ARC-M Recovery Environment for Isaac Lab
Custom environment for training autonomous recovery behaviors in debris fields.
"""

from __future__ import annotations

import torch
from typing import Dict, Tuple, Any
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import robot configurations
from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG

##
# Pre-defined configs
##

@configclass
class DebrisTerrainCfg:
    """Configuration for debris-filled terrain."""
    
    # Terrain type
    terrain_type: str = "debris"
    
    # Ground plane settings
    ground_plane_cfg: sim_utils.GroundPlaneCfg = sim_utils.GroundPlaneCfg()
    
    # Debris settings
    debris_density: float = 0.3  # objects per m²
    debris_size_min: float = 0.05  # meters
    debris_size_max: float = 0.3  # meters
    debris_mass_min: float = 0.1  # kg
    debris_mass_max: float = 5.0  # kg
    
    # Area for debris spawning
    spawn_area: Tuple[float, float] = (10.0, 10.0)  # meters
    
    # Debris shapes
    shapes: list = None  # ["box", "cylinder", "sphere"]
    
    def __post_init__(self):
        if self.shapes is None:
            self.shapes = ["box", "cylinder", "sphere"]


@configclass 
class RecoverySceneCfg(InteractiveSceneCfg):
    """Configuration for the recovery training scene."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # Robot (ANYmal-D by default)
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Contact sensors on feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT",
        history_length=3,
        track_air_time=True,
    )
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP Components
##

@configclass
class CommandsCfg:
    """Configuration for velocity commands."""
    
    # Linear velocity range (m/s)
    lin_vel_x: Tuple[float, float] = (-1.0, 2.0)
    lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
    
    # Angular velocity range (rad/s)
    ang_vel_yaw: Tuple[float, float] = (-1.5, 1.5)
    
    # Heading command
    heading_range: Tuple[float, float] = (-3.14, 3.14)
    
    # Resampling interval (timesteps)
    resampling_time_range: Tuple[float, float] = (10.0, 10.0)


@configclass
class ActionsCfg:
    """Configuration for robot actions."""
    
    # Joint position action
    joint_pos = {
        "asset_name": "robot",
        "joint_names": [".*"],
        "scale": 0.5,
        "use_default_offset": True,
    }


@configclass
class ObservationsCfg:
    """Configuration for observations."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        
        # Enable observation concatenation
        concatenate_terms: bool = True
        
        # Base velocity
        base_lin_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_lin_vel_b)
        base_ang_vel = ObsTerm(func=lambda env: env.scene["robot"].data.root_ang_vel_b)
        
        # Projected gravity
        projected_gravity = ObsTerm(
            func=lambda env: env.scene["robot"].data.projected_gravity_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # Velocity commands
        velocity_commands = ObsTerm(func=lambda env: env.command_manager.get_command("base_velocity"))
        
        # Joint states
        joint_pos = ObsTerm(
            func=lambda env: env.scene["robot"].data.joint_pos - env.scene["robot"].data.default_joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=lambda env: env.scene["robot"].data.joint_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        
        # Previous actions
        actions = ObsTerm(func=lambda env: env.action_manager.action)
        
    # Groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for recovery training."""
    
    # Velocity tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=lambda env: torch.exp(
            -torch.sum(
                torch.square(
                    env.command_manager.get_command("base_velocity")[:, :2]
                    - env.scene["robot"].data.root_lin_vel_b[:, :2]
                ),
                dim=1,
            )
            / 0.25
        ),
        weight=1.5,
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=lambda env: torch.exp(
            -torch.square(
                env.command_manager.get_command("base_velocity")[:, 2]
                - env.scene["robot"].data.root_ang_vel_b[:, 2]
            )
            / 0.25
        ),
        weight=0.75,
    )
    
    # Regularization penalties
    lin_vel_z_l2 = RewTerm(
        func=lambda env: torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2]),
        weight=-2.0,
    )
    
    ang_vel_xy_l2 = RewTerm(
        func=lambda env: torch.sum(
            torch.square(env.scene["robot"].data.root_ang_vel_b[:, :2]), dim=1
        ),
        weight=-0.05,
    )
    
    # Energy penalties
    dof_torques_l2 = RewTerm(
        func=lambda env: torch.sum(torch.square(env.scene["robot"].data.applied_torque), dim=1),
        weight=-0.0002,
    )
    
    dof_acc_l2 = RewTerm(
        func=lambda env: torch.sum(torch.square(env.scene["robot"].data.joint_acc), dim=1),
        weight=-2.5e-7,
    )
    
    action_rate_l2 = RewTerm(
        func=lambda env: torch.sum(
            torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
        ),
        weight=-0.01,
    )
    
    # Stability rewards
    flat_orientation_l2 = RewTerm(
        func=lambda env: torch.sum(
            torch.square(env.scene["robot"].data.projected_gravity_b[:, :2]), dim=1
        ),
        weight=-0.5,
    )
    
    # Recovery-specific: alive bonus
    is_alive = RewTerm(
        func=lambda env: torch.ones(env.num_envs, device=env.device),
        weight=0.5,
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""
    
    # Time limit
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length)
    
    # Base contact with ground (fallen)
    base_contact = DoneTerm(
        func=lambda env: torch.any(
            torch.norm(env.scene["contact_forces"].data.net_forces_w[:, :4], dim=-1) > 1.0,
            dim=1,
        ),
        params={"threshold": 1.0},
    )


def reset_robot_state(env, env_ids: torch.Tensor) -> None:
    """Reset robot to random pose with slight orientation variation."""
    robot = env.scene["robot"]
    num_resets = len(env_ids)
    device = env.device
    
    # Random position offset within spawn area
    pos_offset = torch.zeros(num_resets, 3, device=device)
    pos_offset[:, :2] = torch.rand(num_resets, 2, device=device) * 2.0 - 1.0  # ±1m
    pos_offset[:, 2] = 0.0  # Keep on ground
    
    # Small random orientation (yaw only for stability)
    yaw_angles = torch.rand(num_resets, device=device) * 0.5 - 0.25  # ±0.25 rad
    
    # Create quaternions for yaw rotation
    half_yaw = yaw_angles * 0.5
    quat = torch.zeros(num_resets, 4, device=device)
    quat[:, 0] = torch.cos(half_yaw)  # w
    quat[:, 3] = torch.sin(half_yaw)  # z
    
    # Apply reset
    default_state = robot.data.default_root_state[env_ids].clone()
    default_state[:, :3] += pos_offset
    default_state[:, 3:7] = quat
    
    robot.write_root_state_to_sim(default_state, env_ids)
    robot.reset(env_ids)


def randomize_physics_material(env, env_ids: torch.Tensor, friction_range: tuple, restitution_range: tuple) -> None:
    """Randomize ground friction and restitution for domain randomization."""
    # Note: Full material randomization requires USD API access
    # This is a simplified version that could be extended
    num_resets = len(env_ids)
    device = env.device
    
    # Generate random friction and restitution values
    friction = torch.rand(num_resets, device=device) * (friction_range[1] - friction_range[0]) + friction_range[0]
    restitution = torch.rand(num_resets, device=device) * (restitution_range[1] - restitution_range[0]) + restitution_range[0]
    
    # Store for potential use in reward computation
    if not hasattr(env, '_ground_friction'):
        env._ground_friction = torch.ones(env.num_envs, device=device)
        env._ground_restitution = torch.zeros(env.num_envs, device=device)
    
    env._ground_friction[env_ids] = friction
    env._ground_restitution[env_ids] = restitution


def apply_push_disturbance(env, env_ids: torch.Tensor, force_range: tuple) -> None:
    """Apply random push forces to the robot base."""
    robot = env.scene["robot"]
    num_pushes = len(env_ids)
    device = env.device
    
    # Random force direction (horizontal only for realism)
    force_direction = torch.rand(num_pushes, 2, device=device) * 2.0 - 1.0
    force_direction = force_direction / (torch.norm(force_direction, dim=1, keepdim=True) + 1e-6)
    
    # Random force magnitude
    force_magnitude = torch.rand(num_pushes, device=device) * (force_range[1] - force_range[0]) + force_range[0]
    
    # Construct 3D force vector
    forces = torch.zeros(num_pushes, 3, device=device)
    forces[:, :2] = force_direction * force_magnitude.unsqueeze(1)
    
    # Apply external force to robot base
    # Note: Exact API depends on Isaac Lab version
    robot.set_external_force_and_torque(
        forces=forces,
        torques=torch.zeros_like(forces),
        body_ids=torch.zeros(num_pushes, dtype=torch.long, device=device),  # Base body
        env_ids=env_ids
    )


@configclass
class EventsCfg:
    """Event configuration for domain randomization."""
    
    # Reset robot to random pose
    reset_robot = EventTerm(
        func=reset_robot_state,
        mode="reset",
    )
    
    # Randomize physics
    physics_material = EventTerm(
        func=randomize_physics_material,
        mode="reset",
        params={
            "friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.5),
        },
    )
    
    # Random external pushes
    push_robot = EventTerm(
        func=apply_push_disturbance,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "force_range": (0.0, 50.0),
        },
    )


##
# Environment Configuration
##

@configclass
class RecoveryEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the ARC-M recovery environment."""
    
    # Scene
    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # Episode settings
    episode_length_s: float = 20.0
    decimation: int = 4
    
    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=decimation,
    )
    
    # Viewer camera
    viewer = sim_utils.ViewerCfg(
        eye=(3.0, 3.0, 2.0),
        target=(0.0, 0.0, 0.5),
    )


class RecoveryEnv(ManagerBasedEnv):
    """
    ARC-M Recovery Environment.
    
    Trains quadruped robots to recover from stuck/trapped states
    in debris-filled environments.
    """
    
    cfg: RecoveryEnvCfg
    
    def __init__(self, cfg: RecoveryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Track stuck states
        self._stuck_counter = torch.zeros(self.num_envs, device=self.device)
        self._last_position = torch.zeros(self.num_envs, 3, device=self.device)
        
    def _setup_scene(self):
        """Setup the scene with robot and debris."""
        super()._setup_scene()
        
        # Add debris objects (placeholder - implement debris spawning)
        self._spawn_debris()
        
    def _spawn_debris(self):
        """Spawn random debris objects in the environment."""
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim.spawners import shapes
        
        # Debris configuration
        debris_density = 0.3  # objects per m²
        spawn_area = (10.0, 10.0)  # meters
        debris_size_range = (0.05, 0.3)  # meters
        debris_mass_range = (0.1, 5.0)  # kg
        
        # Calculate number of debris objects
        num_debris = int(debris_density * spawn_area[0] * spawn_area[1])
        
        # Spawn debris for each environment
        for debris_idx in range(num_debris):
            # Random position within spawn area
            x = (torch.rand(1).item() - 0.5) * spawn_area[0]
            y = (torch.rand(1).item() - 0.5) * spawn_area[1]
            z = torch.rand(1).item() * 0.1  # Slight height variation
            
            # Random size
            size = torch.rand(1).item() * (debris_size_range[1] - debris_size_range[0]) + debris_size_range[0]
            
            # Random shape selection
            shape_type = torch.randint(0, 3, (1,)).item()
            
            prim_path = f"{{ENV_REGEX_NS}}/debris_{debris_idx}"
            
            if shape_type == 0:  # Box
                spawn_cfg = shapes.CuboidCfg(
                    size=(size, size, size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=torch.rand(1).item() * (debris_mass_range[1] - debris_mass_range[0]) + debris_mass_range[0]
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.4, 0.3),  # Brown/debris color
                    ),
                )
            elif shape_type == 1:  # Cylinder
                spawn_cfg = shapes.CylinderCfg(
                    radius=size / 2,
                    height=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=torch.rand(1).item() * (debris_mass_range[1] - debris_mass_range[0]) + debris_mass_range[0]
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.4, 0.4, 0.4),  # Gray
                    ),
                )
            else:  # Sphere
                spawn_cfg = shapes.SphereCfg(
                    radius=size / 2,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(
                        mass=torch.rand(1).item() * (debris_mass_range[1] - debris_mass_range[0]) + debris_mass_range[0]
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.6, 0.5, 0.4),  # Tan
                    ),
                )
            
            # Spawn the debris object
            try:
                spawn_cfg.func(
                    prim_path=prim_path,
                    cfg=spawn_cfg,
                    translation=(x, y, z),
                )
            except Exception as e:
                # Silently continue if spawning fails (e.g., headless mode)
                pass
        
        # Store debris count for reference
        self._num_debris = num_debris
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        super()._pre_physics_step(actions)
        
        # Track position for stuck detection
        self._last_position = self.scene["robot"].data.root_pos_w.clone()
        
    def _post_physics_step(self):
        """Process after physics step."""
        super()._post_physics_step()
        
        # Detect stuck states
        current_pos = self.scene["robot"].data.root_pos_w
        movement = torch.norm(current_pos - self._last_position, dim=-1)
        
        # Increment counter if barely moving
        stuck_mask = movement < 0.01  # meters per physics step
        self._stuck_counter[stuck_mask] += 1
        self._stuck_counter[~stuck_mask] = 0
        
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get termination signals."""
        terminated, truncated = super()._get_dones()
        
        # Additional termination: stuck too long
        max_stuck_steps = 500  # About 10 seconds at 50Hz
        stuck_termination = self._stuck_counter > max_stuck_steps
        terminated = terminated | stuck_termination
        
        return terminated, truncated


# Register environment with gymnasium
import gymnasium as gym

gym.register(
    id="ARC-M-Recovery-v0",
    entry_point="envs.recovery_env:RecoveryEnv",
    kwargs={
        "cfg": RecoveryEnvCfg(),
    },
)
