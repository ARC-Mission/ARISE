# Reward Engineering Guide for ARC-M

This document provides guidance on designing and refining reward functions using Magistral for autonomous recovery training.

## The Eureka Approach

The ARC-M project implements an LLM-guided reward generation approach inspired by NVIDIA's Eureka research. Instead of manually engineering reward functions, we use Magistral (Mistral's reasoning model) to generate, test, and iteratively refine reward functions.

### Key Principles

1. **Environment as Context**: The LLM receives the raw environment code as context, enabling zero-shot reward generation without task-specific prompt engineering.

2. **Rapid Evaluation**: GPU-accelerated simulation in Isaac Lab allows quick evaluation of large batches of reward candidates.

3. **Evolutionary Search**: Multiple reward candidates are evaluated in parallel, with the best performers informing the next generation.

4. **Reflective Refinement**: Training metrics are fed back to Magistral, which analyzes failures and suggests improvements.

## Prompt Templates

### Initial Reward Generation

```markdown
You are designing a reward function for training a quadruped robot to recover from stuck states.

**Task Description:**
The robot must detect when it's trapped in debris and execute recovery motions (wiggling, pivoting, pushing obstacles) to free itself and resume normal locomotion.

**Available Observations:**
- base_lin_vel: Base linear velocity in robot frame [3D]
- base_ang_vel: Base angular velocity in robot frame [3D]
- projected_gravity: Gravity vector in robot frame [3D]
- commands: Velocity commands [lin_x, lin_y, ang_z]
- dof_pos: Joint positions relative to default [12D]
- dof_vel: Joint velocities [12D]
- contact_forces: Foot contact forces [4D]
- actions: Previous actions [12D]

**Design Guidelines:**
1. Primary objectives should have positive weights (what to achieve)
2. Regularization should have negative weights (what to avoid)
3. Use exponential terms for distance-based rewards: exp(-error² / scale)
4. Include energy penalties to encourage efficient motions
5. Add stability terms to prevent falling during recovery

**Output Format:**
```python
def compute_reward(obs_dict: Dict[str, torch.Tensor], 
                   actions: torch.Tensor, 
                   reset_buf: torch.Tensor) -> torch.Tensor:
    # Your implementation
    return reward  # Shape: (num_envs,)
```
```

### Refinement Prompt

```markdown
The previous reward function produced these training results:

**Metrics (after 500 iterations):**
- Mean reward: {mean_reward}
- Episode length: {episode_length}
- Success rate: {success_rate}
- Torque magnitude: {torque_magnitude}

**Observed Issues:**
{issues_list}

**Analysis Request:**
1. Diagnose the root cause of any problems
2. Suggest specific modifications to the reward function
3. Recommend hyperparameter adjustments if needed
4. Provide confidence level in your recommendations

**Please output an improved reward function.**
```

## Reward Components Library

### Velocity Tracking

```python
# Linear velocity tracking (exponential form)
lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
tracking_lin_vel = torch.exp(-lin_vel_error / 0.25)

# Angular velocity tracking
ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
tracking_ang_vel = torch.exp(-ang_vel_error / 0.25)
```

### Stability Penalties

```python
# Orientation penalty (keep upright)
orientation_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

# Base height penalty (maintain target height)
height_error = torch.square(base_height - target_height)

# Angular velocity penalty (smooth motion)
ang_vel_penalty = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
```

### Energy Efficiency

```python
# Torque penalty
torque_penalty = torch.sum(torch.square(applied_torques), dim=1)

# Acceleration penalty
accel_penalty = torch.sum(torch.square(joint_acc), dim=1)

# Action rate penalty (smooth actions)
action_rate_penalty = torch.sum(torch.square(actions - prev_actions), dim=1)
```

### Recovery-Specific

```python
# Progress toward recovery (simplified example)
velocity_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)
recovery_progress = torch.where(
    is_stuck,
    velocity_magnitude,  # Reward any movement when stuck
    torch.zeros_like(velocity_magnitude)
)

# Debris displacement reward
debris_moved = torch.norm(debris_pos - initial_debris_pos, dim=1)
displacement_reward = torch.clamp(debris_moved, 0, 1.0)

# Foot clearance (avoid dragging)
foot_heights = get_foot_heights(env)
clearance_reward = torch.mean(foot_heights, dim=1)
```

## Common Issues and Solutions

### Issue: Robot Falls Over During Recovery

**Symptoms:**
- Short episode lengths
- High orientation error
- Negative mean reward

**Solutions:**
1. Increase orientation penalty weight
2. Add base height reward
3. Reduce action scale
4. Add smoother action constraints

```python
# Stronger stability terms
orientation_penalty = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * 2.0
fall_penalty = torch.where(is_fallen, torch.ones_like(reward) * -10.0, torch.zeros_like(reward))
```

### Issue: Robot Doesn't Move (Over-Regularized)

**Symptoms:**
- Zero or near-zero velocity
- Low torque magnitude
- Robot stays in place

**Solutions:**
1. Reduce energy penalty weights
2. Increase velocity tracking rewards
3. Add movement bonus when stuck

```python
# Movement incentive
min_velocity_bonus = torch.where(
    torch.norm(base_lin_vel[:, :2], dim=1) > 0.1,
    torch.ones_like(reward) * 0.5,
    torch.zeros_like(reward)
)
```

### Issue: Jittery Motions

**Symptoms:**
- High action rate
- Oscillating joint positions
- Inefficient energy usage

**Solutions:**
1. Increase action rate penalty
2. Add action smoothness term
3. Reduce learning rate

```python
# Stronger smoothness
action_rate_penalty = torch.sum(torch.square(actions - prev_actions), dim=1) * 0.1
action_magnitude_penalty = torch.sum(torch.square(actions), dim=1) * 0.01
```

## Best Practices

1. **Start Simple**: Begin with basic velocity tracking and stability, then add complexity.

2. **Balance Objectives**: No single reward component should dominate. Use TensorBoard to monitor individual reward terms.

3. **Curriculum Learning**: Start with easy scenarios (flat terrain, light debris) and gradually increase difficulty.

4. **Domain Randomization**: Ensure rewards are robust across parameter variations (friction, mass, etc.).

5. **Validation**: Test learned policies in scenarios not seen during training.

## Reward Scales

Typical weight magnitudes for balanced training:

| Component | Weight Range | Notes |
|-----------|-------------|-------|
| Velocity tracking | 1.0 - 2.0 | Primary objective |
| Angular velocity tracking | 0.5 - 1.0 | Secondary |
| Orientation penalty | -0.5 to -2.0 | Critical for stability |
| Torque penalty | -1e-4 to -1e-3 | Small but important |
| Action rate penalty | -0.01 to -0.1 | Depends on desired smoothness |
| Alive bonus | 0.1 - 1.0 | Encourages survival |
| Recovery progress | 1.0 - 3.0 | Task-specific |

## Monitoring and Debugging

### TensorBoard Metrics

Log individual reward components to identify issues:

```python
# In training loop
writer.add_scalar('rewards/tracking_lin_vel', tracking_reward.mean(), step)
writer.add_scalar('rewards/orientation_penalty', orientation_penalty.mean(), step)
writer.add_scalar('rewards/torque_penalty', torque_penalty.mean(), step)
```

### Diagnostic Plots

- Reward distribution histogram
- Episode length over training
- Success rate by difficulty level
- Joint torque profiles
- Foot contact patterns

## References

1. [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/abs/2310.12931)
2. [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
3. [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
4. [Magistral Technical Report](https://arxiv.org/abs/2506.10910)

## Contributing

Contributions are welcome! Please submit a Pull Request.

## License

MIT License
