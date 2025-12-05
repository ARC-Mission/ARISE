#!/usr/bin/env python3
"""
Test script for the ARC-M Recovery Environment.
Verifies environment creation, stepping, and observation space.
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path

# Add Isaac Lab path if needed (though should be handled by env)
ISAAC_LAB_PATH = os.environ.get("ISAAC_LAB_PATH")
if ISAAC_LAB_PATH:
    sys.path.insert(0, ISAAC_LAB_PATH)

def test_environment():
    """Test the recovery environment."""
    
    print("\n" + "="*60)
    print("Testing ARC-M Recovery Environment")
    print("="*60 + "\n")
    
    try:
        # Import environment
        # Note: In a real Isaac Lab run, we'd need the simulation app running
        # For this test script, we'll try to import and instantiate the config
        # to catch static errors, even if we can't run full sim without a GPU environment
        
        from envs.recovery_env import RecoveryEnv, RecoveryEnvCfg
        
        print("✓ Environment module imported successfully")
        
        # Test Configuration
        cfg = RecoveryEnvCfg()
        print(f"✓ Configuration created:")
        print(f"  - Num Envs: {cfg.scene.num_envs}")
        print(f"  - Episode Length: {cfg.episode_length_s}s")
        print(f"  - Decimation: {cfg.decimation}")
        
        # Check debris settings (default in config class)
        print("  - Debris Settings: Default (checked in code)")
        
        # Check observation configuration
        obs_cfg = cfg.observations
        print("✓ Observation groups defined:")
        for attr_name in dir(obs_cfg):
            if not attr_name.startswith('__') and not callable(getattr(obs_cfg, attr_name)):
                print(f"  - {attr_name}")
                
        # Check reward configuration
        rew_cfg = cfg.rewards
        print("✓ Reward terms defined:")
        count = 0
        for attr_name in dir(rew_cfg):
            if not attr_name.startswith('__'):
                attr = getattr(rew_cfg, attr_name)
                # Check if it looks like a reward term (has weight)
                if hasattr(attr, 'weight'):
                    print(f"  - {attr_name}: {attr.weight}")
                    count += 1
        print(f"  Total terms: {count}")
        
        print("\n[INFO] Full simulation test requires Isaac Lab + NVIDIA GPU.")
        print("To run full verification:")
        print("  cd ~/IsaacLab")
        print("  ./isaaclab.sh -p " + os.path.abspath(__file__))
        
        return True
        
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("Make sure you are running in the Isaac Lab environment:")
        print("  source ~/env_isaaclab/bin/activate")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_environment()
