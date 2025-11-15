#!/usr/bin/env python3
"""
MADDPG Inference Script
Used by Go allocator for model predictions
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path

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

        agents.append(actor)

    return agents


def predict(agents, states):
    """Run inference on states"""
    actions = []

    with torch.no_grad():
        for i, agent in enumerate(agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            action = agent(state).squeeze(0).cpu().numpy()
            actions.append(action.tolist())

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

        # Run inference
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
