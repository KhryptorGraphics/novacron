#!/usr/bin/env python3
"""MADDPG Inference Script
Used by Go allocator for model predictions
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from typing import List

# Import from train.py
from train import Actor

def load_agents(model_path, num_agents, state_dim=8, action_dim=4, hidden_dim=256):
    """Load trained MADDPG agents"""
    agents = []
    
    for i in range(num_agents):
        # Create actor network
        actor = Actor(state_dim, action_dim, hidden_dim)
        
        # Load weights
        agent_path = Path(model_path) / f"agent_{i}.pt"
        checkpoint = torch.load(agent_path, map_location='cpu')
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        agents.append(agent)
    
    return agents

def predict_agent(agent, state):
    """Run inference for a single agent"""
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent(state_tensor).squeeze(0).cpu().numpy()
        return action.tolist()

def predict(agents, states):
    """Run inference on states in parallel"""
    actions = []
    
    with Pool() as pool:
        results = pool.starmap(predict_agent, [(agent, state) for agent, state in zip(agents, states)])
        actions = results
    
    return actions

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: inference.py <model_path> <states_file>"}))
        sys.exit(1)
    
    model_path = sys.argv[1]
    states_file = sys.argv[2]
    
    try:
        # Load states
        with open(states_file, 'r') as f:
            data = json.load(f)
            states = data['states']
        
        num_agents = len(states)
        
        # Load agents
        agents = load_agents(model_path, num_agents)
        
        # Run inference in parallel
        actions = predict(agents, states)
        
        # Output results
        result = {
            "actions": actions,
            "num_agents": num_agents
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()