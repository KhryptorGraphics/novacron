#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
import random
import os
import json

# Use absolute import for DistributedResourceEnv
from environment import DistributedResourceEnv

# Use absolute import for MADDPGTrainer
from train import MADDPGTrainer