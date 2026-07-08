#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time

# Import environment directly since it's in the same directory
from environment import DistributedResourceEnv

# Import train directly since it's in the same directory
from train import MADDPGTrainer

if __name__ == "__main__":
    env = DistributedResourceEnv()
    trainer = MADDPGTrainer(env)
    start_time = time.time()
    obs = env.reset()
    for _ in range(10):
        action = trainer.policy(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break
    end_time = time.time()
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")