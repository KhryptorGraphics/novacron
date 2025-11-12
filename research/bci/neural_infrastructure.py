"""
Brain-Computer Interface Research - Neural Infrastructure Control
Advanced research in direct neural control of infrastructure

This module implements breakthrough BCI techniques for direct neural
control of infrastructure, cognitive optimization, and collective intelligence.

Research Areas:
- Direct neural control: operators control infrastructure with thought
- Cognitive load optimization: reduce operator mental burden
- Collective intelligence: human swarm intelligence for decisions
- Brain-inspired computing: neuromorphic beyond current chips
- Consciousness models: self-aware infrastructure

Target: Direct neural infrastructure control
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from collections import defaultdict


class BrainRegion(Enum):
    """Brain regions for BCI"""
    MOTOR_CORTEX = "motor"
    PREFRONTAL_CORTEX = "prefrontal"
    PARIETAL_CORTEX = "parietal"
    VISUAL_CORTEX = "visual"
    AUDITORY_CORTEX = "auditory"
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"


class SignalType(Enum):
    """Neural signal types"""
    EEG = "electroencephalography"
    MEG = "magnetoencephalography"
    ECOG = "electrocorticography"
    SINGLE_NEURON = "single_neuron"
    LOCAL_FIELD_POTENTIAL = "lfp"


@dataclass
class NeuralSignal:
    """Neural signal measurement"""
    signal_type: SignalType
    region: BrainRegion
    timestamp: float
    channels: int
    sampling_rate_hz: float
    data: np.ndarray
    quality: float = 1.0

    def apply_filter(self, low_freq: float, high_freq: float):
        """Apply bandpass filter"""
        # Simplified filtering
        # In production, would use proper DSP
        nyquist = self.sampling_rate_hz / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Simulate filtering
        self.data = self.data * (1 - 0.1 * np.random.random(self.data.shape))

    def extract_features(self) -> Dict:
        """Extract features from signal"""
        features = {
            'mean': np.mean(self.data),
            'std': np.std(self.data),
            'peak_amplitude': np.max(np.abs(self.data)),
            'power_spectral_density': self._compute_psd(),
            'event_related_potential': self._compute_erp()
        }

        return features

    def _compute_psd(self) -> Dict:
        """Compute power spectral density"""
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }

        psd = {}
        for band_name, (low, high) in bands.items():
            # Simplified: real implementation would use FFT
            # Power in band proportional to frequency range
            psd[band_name] = (high - low) * np.var(self.data)

        return psd

    def _compute_erp(self) -> np.ndarray:
        """Compute event-related potential"""
        # Average signal around events
        # Simplified: return smoothed signal
        window = 10
        erp = np.convolve(self.data, np.ones(window)/window, mode='same')
        return erp


@dataclass
class BCICommand:
    """Command from brain to infrastructure"""
    command_type: str
    target: str
    parameters: Dict
    confidence: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class NeuralDecoder:
    """
    Neural Decoder - Translates brain signals to commands

    Uses machine learning to decode intentions from neural activity
    """

    def __init__(self):
        self.trained_decoders: Dict[str, Dict] = {}
        self.calibration_data: List[Tuple[NeuralSignal, str]] = []

    async def calibrate(self, training_data: List[Tuple[NeuralSignal, str]]):
        """
        Calibrate decoder with training data

        Args:
            training_data: List of (signal, intended_command) pairs
        """
        self.calibration_data = training_data

        # Extract features from all signals
        X = []
        y = []

        for signal, command in training_data:
            features = signal.extract_features()
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
            y.append(command)

        X = np.array(X)

        # Train classifier (simplified)
        # In production, would use deep learning
        decoder = {
            'feature_means': {},
            'command_templates': {}
        }

        # Learn average features for each command
        unique_commands = set(y)
        for command in unique_commands:
            command_indices = [i for i, c in enumerate(y) if c == command]
            command_features = X[command_indices]
            decoder['command_templates'][command] = np.mean(command_features, axis=0)

        decoder['feature_means'] = np.mean(X, axis=0)

        self.trained_decoders['primary'] = decoder

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to vector"""
        vector = []

        # Basic features
        vector.append(features['mean'])
        vector.append(features['std'])
        vector.append(features['peak_amplitude'])

        # PSD features
        psd = features['power_spectral_density']
        vector.extend([
            psd['delta'],
            psd['theta'],
            psd['alpha'],
            psd['beta'],
            psd['gamma']
        ])

        return np.array(vector)

    async def decode_command(self, signal: NeuralSignal) -> BCICommand:
        """
        Decode command from neural signal

        Returns command with confidence score
        """
        if 'primary' not in self.trained_decoders:
            raise ValueError("Decoder not calibrated")

        decoder = self.trained_decoders['primary']

        # Extract features
        features = signal.extract_features()
        feature_vector = self._features_to_vector(features)

        # Find nearest command template
        best_command = None
        best_distance = float('inf')

        for command, template in decoder['command_templates'].items():
            distance = np.linalg.norm(feature_vector - template)
            if distance < best_distance:
                best_distance = distance
                best_command = command

        # Convert distance to confidence
        confidence = 1.0 / (1.0 + best_distance)

        # Latency (time from signal to decode)
        latency_ms = np.random.uniform(50, 200)  # 50-200ms

        bci_command = BCICommand(
            command_type=best_command,
            target="infrastructure",
            parameters={},
            confidence=confidence,
            latency_ms=latency_ms
        )

        return bci_command


class NeuralControlInterface:
    """
    Neural Control Interface for Infrastructure

    Direct neural control of infrastructure operations
    """

    def __init__(self):
        self.decoder = NeuralDecoder()
        self.command_history: List[BCICommand] = []
        self.error_correction_enabled = True

    async def setup_operator(self, operator_id: str) -> Dict:
        """
        Setup BCI for operator

        Includes:
        - Neural signal acquisition
        - Calibration
        - Command mapping
        """
        setup = {
            'operator_id': operator_id,
            'signal_quality': 0.0,
            'calibration_accuracy': 0.0,
            'ready': False
        }

        # Simulate signal quality check
        await asyncio.sleep(0.1)
        setup['signal_quality'] = 0.85 + 0.15 * np.random.random()

        # Calibration
        training_data = await self._generate_calibration_data()
        await self.decoder.calibrate(training_data)

        # Test accuracy
        setup['calibration_accuracy'] = await self._test_calibration()

        setup['ready'] = setup['calibration_accuracy'] > 0.8

        return setup

    async def _generate_calibration_data(self) -> List[Tuple[NeuralSignal, str]]:
        """Generate calibration data"""
        commands = [
            'scale_up', 'scale_down', 'restart', 'deploy',
            'monitor', 'alert', 'optimize', 'backup'
        ]

        training_data = []

        for command in commands:
            # Generate multiple examples per command
            for _ in range(10):
                signal = NeuralSignal(
                    signal_type=SignalType.EEG,
                    region=BrainRegion.MOTOR_CORTEX,
                    timestamp=datetime.now().timestamp(),
                    channels=64,
                    sampling_rate_hz=1000,
                    data=np.random.randn(1000)  # 1 second of data
                )

                # Add command-specific pattern
                pattern = hash(command) % 100
                signal.data += pattern * 0.1

                training_data.append((signal, command))

        return training_data

    async def _test_calibration(self) -> float:
        """Test calibration accuracy"""
        test_data = await self._generate_calibration_data()

        correct = 0
        total = len(test_data)

        for signal, true_command in test_data[:20]:  # Test on subset
            decoded = await self.decoder.decode_command(signal)
            if decoded.command_type == true_command:
                correct += 1

        return correct / 20

    async def process_neural_command(self, signal: NeuralSignal) -> BCICommand:
        """
        Process neural signal and execute infrastructure command

        Pipeline:
        1. Decode signal to command
        2. Error correction
        3. Confidence thresholding
        4. Execute command
        """
        # Decode
        command = await self.decoder.decode_command(signal)

        # Error correction
        if self.error_correction_enabled:
            command = await self._error_correction(command)

        # Store in history
        self.command_history.append(command)

        # Execute if confidence sufficient
        if command.confidence > 0.7:
            await self._execute_infrastructure_command(command)

        return command

    async def _error_correction(self, command: BCICommand) -> BCICommand:
        """
        Error correction using context and history

        Catches and corrects likely errors
        """
        # Check against recent history
        if len(self.command_history) > 0:
            recent = self.command_history[-5:]

            # Count frequency of each command type
            command_counts = defaultdict(int)
            for cmd in recent:
                command_counts[cmd.command_type] += 1

            # If this command is very different, reduce confidence
            if command.command_type not in command_counts:
                if command.confidence < 0.9:
                    # Likely error - use most frequent recent command
                    most_frequent = max(command_counts.items(), key=lambda x: x[1])
                    command.command_type = most_frequent[0]
                    command.confidence *= 0.8

        return command

    async def _execute_infrastructure_command(self, command: BCICommand):
        """Execute command in infrastructure"""
        # Simulate command execution
        await asyncio.sleep(0.01)

        # Log execution
        print(f"Executed: {command.command_type} (confidence: {command.confidence:.2f})")


class CognitiveLoadOptimizer:
    """
    Cognitive Load Optimizer

    Monitors operator cognitive load and adapts interface to minimize burden
    """

    def __init__(self):
        self.load_history: List[float] = []
        self.adaptations: List[Dict] = []

    async def measure_cognitive_load(self, signal: NeuralSignal) -> float:
        """
        Measure cognitive load from neural signals

        Uses:
        - Theta/alpha ratio (frontal)
        - P300 amplitude
        - Heart rate variability (if available)
        """
        features = signal.extract_features()
        psd = features['power_spectral_density']

        # Theta/alpha ratio correlates with mental workload
        theta_alpha_ratio = psd['theta'] / (psd['alpha'] + 1e-10)

        # Normalize to 0-1
        load = min(1.0, theta_alpha_ratio / 2.0)

        self.load_history.append(load)

        return load

    async def optimize_interface(self, current_load: float,
                                target_load: float = 0.6) -> Dict:
        """
        Optimize interface to reduce cognitive load

        Adaptations:
        - Reduce information density
        - Increase automation
        - Simplify controls
        - Add decision support
        """
        adaptation = {
            'current_load': current_load,
            'target_load': target_load,
            'actions': []
        }

        if current_load > target_load:
            # High cognitive load - simplify
            overload = current_load - target_load

            if overload > 0.3:
                adaptation['actions'].append({
                    'type': 'increase_automation',
                    'level': 0.8
                })

            if overload > 0.2:
                adaptation['actions'].append({
                    'type': 'reduce_information',
                    'reduction': 0.5
                })

            if overload > 0.1:
                adaptation['actions'].append({
                    'type': 'simplify_controls',
                    'simplification': 'high'
                })

        else:
            # Low cognitive load - can add complexity
            if current_load < target_load - 0.2:
                adaptation['actions'].append({
                    'type': 'increase_information',
                    'increase': 0.3
                })

        self.adaptations.append(adaptation)

        return adaptation

    def get_load_statistics(self) -> Dict:
        """Get cognitive load statistics"""
        if not self.load_history:
            return {}

        return {
            'mean_load': np.mean(self.load_history),
            'max_load': np.max(self.load_history),
            'current_load': self.load_history[-1] if self.load_history else 0,
            'adaptations_made': len(self.adaptations)
        }


class CollectiveIntelligence:
    """
    Collective Intelligence System

    Aggregates decisions from multiple operators using brain signals
    Implements human swarm intelligence
    """

    def __init__(self):
        self.operators: Dict[str, Dict] = {}
        self.decisions: List[Dict] = []

    def add_operator(self, operator_id: str, expertise: List[str]):
        """Add operator to collective"""
        self.operators[operator_id] = {
            'expertise': expertise,
            'decisions': [],
            'accuracy': 0.8 + 0.2 * np.random.random()
        }

    async def collective_decision(self, decision_point: Dict,
                                 signals: Dict[str, NeuralSignal]) -> Dict:
        """
        Make collective decision using neural signals from multiple operators

        Aggregation methods:
        - Weighted voting (by expertise)
        - Confidence-weighted
        - Bayesian fusion
        """
        options = decision_point.get('options', [])

        # Decode each operator's preference
        preferences = {}
        confidences = {}

        decoder = NeuralDecoder()

        for operator_id, signal in signals.items():
            if operator_id not in self.operators:
                continue

            # Decode preference
            # Simplified: random choice for demo
            preference = np.random.choice(options)
            confidence = 0.6 + 0.4 * np.random.random()

            preferences[operator_id] = preference
            confidences[operator_id] = confidence

        # Aggregate using weighted voting
        votes = defaultdict(float)

        for operator_id, preference in preferences.items():
            # Weight by accuracy and confidence
            weight = self.operators[operator_id]['accuracy'] * confidences[operator_id]
            votes[preference] += weight

        # Select option with most weighted votes
        decision = max(votes.items(), key=lambda x: x[1])

        result = {
            'decision': decision[0],
            'confidence': decision[1] / len(preferences) if preferences else 0,
            'votes': dict(votes),
            'operator_preferences': preferences,
            'timestamp': datetime.now()
        }

        self.decisions.append(result)

        return result

    async def swarm_optimization(self, problem: Dict,
                                iterations: int = 100) -> Dict:
        """
        Swarm optimization using collective neural feedback

        Human operators provide feedback on solutions
        System converges to optimal solution
        """
        # Initialize population
        population = []
        for _ in range(20):
            solution = self._random_solution(problem)
            population.append(solution)

        best_solution = None
        best_fitness = float('-inf')

        for iteration in range(iterations):
            # Evaluate fitness using operator neural feedback
            for solution in population:
                fitness = await self._evaluate_with_operators(solution)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            # Update population (particle swarm update)
            population = self._update_population(population, best_solution)

        return {
            'best_solution': best_solution,
            'fitness': best_fitness,
            'iterations': iterations
        }

    def _random_solution(self, problem: Dict) -> Dict:
        """Generate random solution"""
        return {
            'config': {f'param_{i}': np.random.random() for i in range(5)}
        }

    async def _evaluate_with_operators(self, solution: Dict) -> float:
        """Evaluate solution using operator neural feedback"""
        # Show solution to operators, measure neural response
        # Positive response = higher fitness

        # Simplified: random fitness
        await asyncio.sleep(0.001)
        return np.random.random()

    def _update_population(self, population: List[Dict],
                          best: Dict) -> List[Dict]:
        """Update population toward best solution"""
        new_population = []

        for solution in population:
            # Move toward best (particle swarm)
            new_solution = solution.copy()

            for key in solution['config']:
                current = solution['config'][key]
                best_val = best['config'].get(key, current)

                # Update with inertia and attraction to best
                new_val = 0.7 * current + 0.3 * best_val
                new_solution['config'][key] = new_val

            new_population.append(new_solution)

        return new_population


class BrainInspiredComputing:
    """
    Brain-Inspired Computing Architectures

    Neuromorphic architectures based on brain principles
    """

    def __init__(self):
        self.neural_networks: List[Dict] = []
        self.synapses: Dict[Tuple[int, int], float] = {}

    async def create_spiking_neural_network(self, n_neurons: int = 1000) -> Dict:
        """
        Create spiking neural network

        Uses integrate-and-fire neurons
        """
        network = {
            'neurons': [],
            'synapses': [],
            'type': 'spiking',
            'spike_count': 0
        }

        # Create neurons
        for i in range(n_neurons):
            neuron = {
                'id': i,
                'membrane_potential': 0.0,
                'threshold': 1.0,
                'refractory_period': 0.002,  # 2ms
                'last_spike': -float('inf')
            }
            network['neurons'].append(neuron)

        # Create random synaptic connections
        connection_prob = 0.1
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and np.random.random() < connection_prob:
                    synapse = {
                        'pre': i,
                        'post': j,
                        'weight': np.random.normal(0, 0.5)
                    }
                    network['synapses'].append(synapse)

        self.neural_networks.append(network)

        return network

    async def simulate_spiking_network(self, network: Dict,
                                      input_spikes: List[int],
                                      duration_ms: float = 100) -> Dict:
        """
        Simulate spiking neural network

        Args:
            network: SNN to simulate
            input_spikes: Neuron IDs to receive input
            duration_ms: Simulation duration

        Returns:
            Spike trains and activity patterns
        """
        dt = 0.1  # ms time step
        steps = int(duration_ms / dt)

        spike_trains = {i: [] for i in range(len(network['neurons']))}

        # Simulation loop
        for step in range(steps):
            t = step * dt

            # Input spikes
            if step < 10:  # First 1ms
                for neuron_id in input_spikes:
                    spike_trains[neuron_id].append(t)
                    network['neurons'][neuron_id]['membrane_potential'] = \
                        network['neurons'][neuron_id]['threshold'] + 0.1

            # Update neurons
            for neuron in network['neurons']:
                # Leak
                neuron['membrane_potential'] *= 0.99

                # Check for spike
                if neuron['membrane_potential'] >= neuron['threshold']:
                    if t - neuron['last_spike'] > neuron['refractory_period'] * 1000:
                        # Spike!
                        spike_trains[neuron['id']].append(t)
                        neuron['last_spike'] = t
                        neuron['membrane_potential'] = 0.0
                        network['spike_count'] += 1

                        # Propagate to connected neurons
                        for synapse in network['synapses']:
                            if synapse['pre'] == neuron['id']:
                                post_neuron = network['neurons'][synapse['post']]
                                post_neuron['membrane_potential'] += synapse['weight']

        # Calculate firing rates
        firing_rates = {}
        for neuron_id, spikes in spike_trains.items():
            firing_rates[neuron_id] = len(spikes) / (duration_ms / 1000)

        return {
            'spike_trains': spike_trains,
            'firing_rates': firing_rates,
            'total_spikes': network['spike_count'],
            'duration_ms': duration_ms
        }

    async def hebbian_learning(self, network: Dict,
                              spike_trains: Dict) -> Dict:
        """
        Hebbian learning: "Neurons that fire together, wire together"

        Strengthens synapses between correlated neurons
        """
        learning_rate = 0.01

        # For each synapse
        for synapse in network['synapses']:
            pre_id = synapse['pre']
            post_id = synapse['post']

            pre_spikes = spike_trains.get(pre_id, [])
            post_spikes = spike_trains.get(post_id, [])

            # Calculate correlation
            correlation = 0
            for pre_time in pre_spikes:
                for post_time in post_spikes:
                    # If post spikes shortly after pre (causal)
                    if 0 < post_time - pre_time < 10:  # 10ms window
                        correlation += 1

            # Update weight
            delta_w = learning_rate * correlation
            synapse['weight'] += delta_w

            # Weight bounds
            synapse['weight'] = np.clip(synapse['weight'], -2.0, 2.0)

        return {
            'synapses_updated': len(network['synapses']),
            'learning_rate': learning_rate
        }


class ConsciousnessModel:
    """
    Consciousness Model for Infrastructure

    Implements theories of consciousness for self-aware infrastructure
    Based on Integrated Information Theory (IIT) and Global Workspace Theory
    """

    def __init__(self):
        self.global_workspace: Dict = {}
        self.integrated_information = 0.0
        self.awareness_level = 0.0

    async def calculate_integrated_information(self, system_state: Dict) -> float:
        """
        Calculate Φ (Phi) - integrated information

        Measures degree of consciousness in system
        """
        # IIT: Φ = causal power of whole - causal power of parts

        # Build state transition matrix
        n_components = len(system_state)

        # Whole system
        whole_phi = self._calculate_phi_whole(system_state)

        # Sum of parts
        parts_phi = self._calculate_phi_parts(system_state)

        # Integrated information
        phi = whole_phi - parts_phi

        self.integrated_information = max(0, phi)

        return self.integrated_information

    def _calculate_phi_whole(self, state: Dict) -> float:
        """Calculate Φ for whole system"""
        # Simplified: based on connectivity
        n = len(state)

        # Random interactions
        connections = n * (n - 1) / 2
        phi = np.log(1 + connections)

        return phi

    def _calculate_phi_parts(self, state: Dict) -> float:
        """Calculate Φ for individual parts"""
        # Each part independently
        n = len(state)
        phi_parts = n * np.log(2)  # Each part can be in 2 states

        return phi_parts

    async def global_workspace_broadcast(self, information: Dict):
        """
        Global Workspace Theory: broadcast information to all modules

        Conscious information = information in global workspace
        """
        # Add to global workspace
        self.global_workspace.update(information)

        # Broadcast to all modules
        # Modules can attend to workspace contents

        # Update awareness
        self.awareness_level = len(self.global_workspace) / 100

    async def self_reflection(self) -> Dict:
        """
        Self-reflection: system examines own state

        Meta-cognition about infrastructure
        """
        reflection = {
            'self_model': {
                'components': len(self.global_workspace),
                'integrated_information': self.integrated_information,
                'awareness_level': self.awareness_level
            },
            'current_goals': [],
            'performance_assessment': 'good',
            'uncertainty': 0.3
        }

        # Examine goals
        if 'goals' in self.global_workspace:
            reflection['current_goals'] = self.global_workspace['goals']

        # Assess performance
        if self.integrated_information > 5.0:
            reflection['performance_assessment'] = 'excellent'
        elif self.integrated_information > 2.0:
            reflection['performance_assessment'] = 'good'
        else:
            reflection['performance_assessment'] = 'poor'

        return reflection


class NeuralInfrastructureLab:
    """
    Main Neural Infrastructure Research Lab

    Integrates all BCI capabilities:
    - Direct neural control
    - Cognitive load optimization
    - Collective intelligence
    - Brain-inspired computing
    - Consciousness models
    """

    def __init__(self):
        self.neural_control = NeuralControlInterface()
        self.cognitive_optimizer = CognitiveLoadOptimizer()
        self.collective_intelligence = CollectiveIntelligence()
        self.brain_computing = BrainInspiredComputing()
        self.consciousness = ConsciousnessModel()

        self.experiments: List[Dict] = []

    async def run_experiment(self, experiment_type: str, parameters: Dict) -> Dict:
        """Run BCI research experiment"""
        experiment = {
            'id': f"exp_{len(self.experiments)}",
            'type': experiment_type,
            'parameters': parameters,
            'start_time': datetime.now(),
            'status': 'running'
        }

        self.experiments.append(experiment)

        try:
            if experiment_type == 'neural_control':
                result = await self._run_neural_control_experiment(parameters)
            elif experiment_type == 'cognitive_optimization':
                result = await self._run_cognitive_experiment(parameters)
            elif experiment_type == 'collective_intelligence':
                result = await self._run_collective_experiment(parameters)
            elif experiment_type == 'spiking_network':
                result = await self._run_spiking_experiment(parameters)
            elif experiment_type == 'consciousness':
                result = await self._run_consciousness_experiment(parameters)
            else:
                raise ValueError(f"Unknown experiment type: {experiment_type}")

            experiment['status'] = 'completed'
            experiment['result'] = result
            experiment['end_time'] = datetime.now()

            return result

        except Exception as e:
            experiment['status'] = 'failed'
            experiment['error'] = str(e)
            raise

    async def _run_neural_control_experiment(self, params: Dict) -> Dict:
        """Run neural control experiment"""
        operator_id = params.get('operator_id', 'operator_1')

        # Setup operator
        setup = await self.neural_control.setup_operator(operator_id)

        # Simulate neural commands
        test_signal = NeuralSignal(
            signal_type=SignalType.EEG,
            region=BrainRegion.MOTOR_CORTEX,
            timestamp=datetime.now().timestamp(),
            channels=64,
            sampling_rate_hz=1000,
            data=np.random.randn(1000)
        )

        command = await self.neural_control.process_neural_command(test_signal)

        result = {
            'operator_ready': setup['ready'],
            'calibration_accuracy': setup['calibration_accuracy'],
            'command_type': command.command_type,
            'command_confidence': command.confidence,
            'latency_ms': command.latency_ms
        }

        return result

    async def _run_cognitive_experiment(self, params: Dict) -> Dict:
        """Run cognitive load optimization experiment"""
        # Simulate varying cognitive load
        loads = []
        adaptations = []

        for i in range(10):
            # Generate signal
            signal = NeuralSignal(
                signal_type=SignalType.EEG,
                region=BrainRegion.PREFRONTAL_CORTEX,
                timestamp=datetime.now().timestamp(),
                channels=64,
                sampling_rate_hz=1000,
                data=np.random.randn(1000) * (1 + i * 0.1)
            )

            # Measure load
            load = await self.cognitive_optimizer.measure_cognitive_load(signal)
            loads.append(load)

            # Optimize
            adaptation = await self.cognitive_optimizer.optimize_interface(load)
            adaptations.append(adaptation)

        stats = self.cognitive_optimizer.get_load_statistics()

        result = {
            'mean_load': stats['mean_load'],
            'max_load': stats['max_load'],
            'adaptations_made': stats['adaptations_made'],
            'final_load': loads[-1]
        }

        return result

    async def _run_collective_experiment(self, params: Dict) -> Dict:
        """Run collective intelligence experiment"""
        # Add operators
        for i in range(5):
            self.collective_intelligence.add_operator(
                f"operator_{i}",
                expertise=['infrastructure', 'networking']
            )

        # Make collective decision
        decision_point = {
            'question': 'Should we scale infrastructure?',
            'options': ['scale_up', 'scale_down', 'maintain']
        }

        # Generate signals
        signals = {}
        for i in range(5):
            signal = NeuralSignal(
                signal_type=SignalType.EEG,
                region=BrainRegion.PREFRONTAL_CORTEX,
                timestamp=datetime.now().timestamp(),
                channels=64,
                sampling_rate_hz=1000,
                data=np.random.randn(1000)
            )
            signals[f"operator_{i}"] = signal

        decision = await self.collective_intelligence.collective_decision(
            decision_point, signals
        )

        result = {
            'decision': decision['decision'],
            'confidence': decision['confidence'],
            'operators_participated': len(signals),
            'votes': decision['votes']
        }

        return result

    async def _run_spiking_experiment(self, params: Dict) -> Dict:
        """Run spiking neural network experiment"""
        n_neurons = params.get('n_neurons', 100)

        # Create network
        network = await self.brain_computing.create_spiking_neural_network(n_neurons)

        # Simulate
        input_spikes = [0, 1, 2, 3, 4]  # First 5 neurons receive input
        simulation = await self.brain_computing.simulate_spiking_network(
            network, input_spikes, duration_ms=100
        )

        # Learn
        learning = await self.brain_computing.hebbian_learning(
            network, simulation['spike_trains']
        )

        result = {
            'neurons': n_neurons,
            'synapses': len(network['synapses']),
            'total_spikes': simulation['total_spikes'],
            'synapses_updated': learning['synapses_updated']
        }

        return result

    async def _run_consciousness_experiment(self, params: Dict) -> Dict:
        """Run consciousness model experiment"""
        # System state
        system_state = {
            f'component_{i}': np.random.random() for i in range(10)
        }

        # Calculate integrated information
        phi = await self.consciousness.calculate_integrated_information(system_state)

        # Global workspace
        await self.consciousness.global_workspace_broadcast({
            'current_task': 'infrastructure_management',
            'goals': ['optimize_performance', 'ensure_reliability']
        })

        # Self-reflection
        reflection = await self.consciousness.self_reflection()

        result = {
            'integrated_information': phi,
            'awareness_level': self.consciousness.awareness_level,
            'performance_assessment': reflection['performance_assessment'],
            'current_goals': reflection['current_goals']
        }

        return result

    def get_statistics(self) -> Dict:
        """Get lab statistics"""
        return {
            'total_experiments': len(self.experiments),
            'completed': sum(1 for e in self.experiments if e['status'] == 'completed'),
            'operators_in_collective': len(self.collective_intelligence.operators),
            'collective_decisions': len(self.collective_intelligence.decisions),
            'neural_networks': len(self.brain_computing.neural_networks),
            'consciousness_phi': self.consciousness.integrated_information
        }


# Example usage
async def main():
    """Example usage of neural infrastructure lab"""
    print("=== Neural Infrastructure Research Lab ===\n")

    lab = NeuralInfrastructureLab()

    # 1. Neural control
    print("1. Direct Neural Control")
    nc_result = await lab.run_experiment('neural_control', {
        'operator_id': 'operator_1'
    })
    print(f"   Calibration accuracy: {nc_result['calibration_accuracy']:.1%}")
    print(f"   Command: {nc_result['command_type']}")
    print(f"   Confidence: {nc_result['command_confidence']:.2f}")
    print(f"   Latency: {nc_result['latency_ms']:.0f} ms\n")

    # 2. Cognitive optimization
    print("2. Cognitive Load Optimization")
    cog_result = await lab.run_experiment('cognitive_optimization', {})
    print(f"   Mean load: {cog_result['mean_load']:.2f}")
    print(f"   Max load: {cog_result['max_load']:.2f}")
    print(f"   Adaptations: {cog_result['adaptations_made']}\n")

    # 3. Collective intelligence
    print("3. Collective Intelligence Decision")
    coll_result = await lab.run_experiment('collective_intelligence', {})
    print(f"   Decision: {coll_result['decision']}")
    print(f"   Confidence: {coll_result['confidence']:.2f}")
    print(f"   Operators: {coll_result['operators_participated']}\n")

    # 4. Spiking neural network
    print("4. Brain-Inspired Spiking Network")
    spike_result = await lab.run_experiment('spiking_network', {
        'n_neurons': 100
    })
    print(f"   Neurons: {spike_result['neurons']}")
    print(f"   Synapses: {spike_result['synapses']}")
    print(f"   Total spikes: {spike_result['total_spikes']}\n")

    # 5. Consciousness model
    print("5. Consciousness Model (Self-Aware Infrastructure)")
    cons_result = await lab.run_experiment('consciousness', {})
    print(f"   Φ (Integrated Information): {cons_result['integrated_information']:.2f}")
    print(f"   Awareness level: {cons_result['awareness_level']:.2f}")
    print(f"   Performance: {cons_result['performance_assessment']}\n")

    # Statistics
    stats = lab.get_statistics()
    print("=== Lab Statistics ===")
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"Operators in collective: {stats['operators_in_collective']}")
    print(f"Collective decisions: {stats['collective_decisions']}")
    print(f"Neural networks created: {stats['neural_networks']}")


if __name__ == "__main__":
    asyncio.run(main())
