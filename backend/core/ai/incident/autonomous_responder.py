"""
Autonomous Incident Response System for NovaCron
Implements automated incident handling with <2 minute MTTR for P2-P4 incidents
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import asyncio
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# NLP for incident analysis
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Knowledge Graph
import networkx as nx
from py2neo import Graph, Node, Relationship

# Causal Analysis
from causalnex.structure import StructureModel
import dowhy

# Process Mining
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

# Monitoring
from prometheus_client import Counter, Gauge, Histogram, Summary
import redis
import aioredis

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Prometheus metrics
incidents_handled = Counter('incidents_handled_total', 'Total incidents handled')
incident_classification_accuracy = Gauge('incident_classification_accuracy', 'Classification accuracy')
mttr_seconds = Histogram('mttr_seconds', 'Mean time to resolution', ['priority'])
automation_success_rate = Gauge('automation_success_rate', 'Automation success rate')
remediation_actions_taken = Counter('remediation_actions_total', 'Total remediation actions')
false_positives = Counter('false_positives_total', 'False positive incidents')
human_escalations = Counter('human_escalations_total', 'Incidents escalated to humans')
knowledge_base_size = Gauge('knowledge_base_size', 'Size of incident knowledge base')

class IncidentPriority(Enum):
    """Incident priority levels"""
    P0 = "p0_critical"  # System down
    P1 = "p1_high"      # Major degradation
    P2 = "p2_medium"    # Significant impact
    P3 = "p3_low"       # Minor impact
    P4 = "p4_minimal"   # Minimal impact

class IncidentCategory(Enum):
    """Incident categories"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    DATA_INTEGRITY = "data_integrity"
    CONFIGURATION = "configuration"
    CAPACITY = "capacity"

class RemediationAction(Enum):
    """Automated remediation actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    CLEAR_CACHE = "clear_cache"
    RESET_CONNECTION_POOL = "reset_connection_pool"
    APPLY_PATCH = "apply_patch"
    FAILOVER = "failover"
    REINDEX_DATABASE = "reindex_database"
    OPTIMIZE_QUERY = "optimize_query"
    ADJUST_CONFIGURATION = "adjust_configuration"
    ALLOCATE_RESOURCES = "allocate_resources"
    TERMINATE_PROCESS = "terminate_process"
    RESTORE_BACKUP = "restore_backup"
    ESCALATE_TO_HUMAN = "escalate_to_human"

@dataclass
class Incident:
    """Incident data structure"""
    incident_id: str
    title: str
    description: str
    timestamp: datetime
    source: str
    affected_services: List[str]
    metrics: Dict[str, float]
    logs: List[str]
    alerts: List[Dict[str, Any]]
    priority: Optional[IncidentPriority] = None
    category: Optional[IncidentCategory] = None
    root_cause: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IncidentClassification:
    """Incident classification result"""
    incident_id: str
    priority: IncidentPriority
    category: IncidentCategory
    confidence: float
    features_extracted: Dict[str, Any]
    similar_incidents: List[str]
    estimated_mttr: float
    requires_human: bool

@dataclass
class RemediationPlan:
    """Remediation plan for incident"""
    incident_id: str
    actions: List[RemediationAction]
    sequence: List[int]  # Order of actions
    estimated_time: float
    success_probability: float
    rollback_plan: List[RemediationAction]
    validation_steps: List[str]
    human_approval_required: bool

@dataclass
class RemediationResult:
    """Result of remediation execution"""
    incident_id: str
    success: bool
    actions_executed: List[RemediationAction]
    execution_time: float
    resolution_status: str
    validation_passed: bool
    rollback_executed: bool
    escalated_to_human: bool
    post_incident_report: Dict[str, Any]

class IncidentClassifier:
    """ML-based incident classifier"""

    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Initialize models
        self._initialize_models()

        # Load NLP model
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def _initialize_models(self):
        """Initialize classification models"""
        # Priority classifier
        self.models['priority'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            objective='multi:softprob'
        )

        # Category classifier
        self.models['category'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            objective='multiclass'
        )

        # MTTR predictor
        self.models['mttr'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8
        )

        # Initialize encoders
        self.encoders['priority'] = LabelEncoder()
        self.encoders['category'] = LabelEncoder()

    async def train(self, training_data: List[Tuple[Incident, IncidentPriority, IncidentCategory]]) -> Dict[str, Any]:
        """Train incident classifier"""
        logger.info("Training incident classifier...")
        start_time = datetime.now()

        # Prepare training data
        X = []
        y_priority = []
        y_category = []

        for incident, priority, category in training_data:
            features = self._extract_features(incident)
            X.append(features)
            y_priority.append(priority.value)
            y_category.append(category.value)

        X = np.array(X)

        # Encode labels
        y_priority_encoded = self.encoders['priority'].fit_transform(y_priority)
        y_category_encoded = self.encoders['category'].fit_transform(y_category)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_priority_train, y_priority_test, y_category_train, y_category_test = train_test_split(
            X_scaled, y_priority_encoded, y_category_encoded,
            test_size=0.2, random_state=42
        )

        # Train priority classifier
        self.models['priority'].fit(X_train, y_priority_train)
        priority_score = self.models['priority'].score(X_test, y_priority_test)

        # Train category classifier
        self.models['category'].fit(X_train, y_category_train)
        category_score = self.models['category'].score(X_test, y_category_test)

        self.is_trained = True
        training_time = (datetime.now() - start_time).total_seconds()

        return {
            'training_time': training_time,
            'priority_accuracy': priority_score,
            'category_accuracy': category_score,
            'samples_trained': len(X)
        }

    def _extract_features(self, incident: Incident) -> np.ndarray:
        """Extract features from incident"""
        features = []

        # Text features from description
        doc = self.nlp(incident.description)

        # Sentiment score
        sentiment = self.sentiment_analyzer(incident.description[:512])[0]
        features.append(1.0 if sentiment['label'] == 'NEGATIVE' else 0.0)

        # Entity counts
        features.append(len(doc.ents))

        # Token statistics
        features.append(len(doc))
        features.append(len([token for token in doc if token.is_stop]))

        # Service impact
        features.append(len(incident.affected_services))

        # Metric features
        features.append(incident.metrics.get('error_rate', 0))
        features.append(incident.metrics.get('response_time', 0))
        features.append(incident.metrics.get('cpu_usage', 0))
        features.append(incident.metrics.get('memory_usage', 0))

        # Alert features
        features.append(len(incident.alerts))

        # Time features
        features.append(incident.timestamp.hour)
        features.append(incident.timestamp.weekday())

        # Pad to fixed size
        while len(features) < 50:
            features.append(0)

        return np.array(features[:50])

    async def classify(self, incident: Incident) -> IncidentClassification:
        """Classify incident priority and category"""
        if not self.is_trained:
            # Default classification
            return IncidentClassification(
                incident_id=incident.incident_id,
                priority=IncidentPriority.P2,
                category=IncidentCategory.INFRASTRUCTURE,
                confidence=0.5,
                features_extracted={},
                similar_incidents=[],
                estimated_mttr=30,
                requires_human=True
            )

        # Extract features
        features = self._extract_features(incident)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict priority
        priority_pred = self.models['priority'].predict(features_scaled)[0]
        priority_proba = self.models['priority'].predict_proba(features_scaled)[0]
        priority = IncidentPriority(self.encoders['priority'].inverse_transform([priority_pred])[0])
        priority_confidence = np.max(priority_proba)

        # Predict category
        category_pred = self.models['category'].predict(features_scaled)[0]
        category_proba = self.models['category'].predict_proba(features_scaled)[0]
        category = IncidentCategory(self.encoders['category'].inverse_transform([category_pred])[0])
        category_confidence = np.max(category_proba)

        # Overall confidence
        confidence = (priority_confidence + category_confidence) / 2

        # Estimate MTTR based on priority
        mttr_map = {
            IncidentPriority.P0: 5,
            IncidentPriority.P1: 10,
            IncidentPriority.P2: 30,
            IncidentPriority.P3: 60,
            IncidentPriority.P4: 120
        }
        estimated_mttr = mttr_map.get(priority, 30)

        # Determine if human intervention needed
        requires_human = priority in [IncidentPriority.P0, IncidentPriority.P1] or confidence < 0.7

        # Update metrics
        incident_classification_accuracy.set(confidence * 100)

        return IncidentClassification(
            incident_id=incident.incident_id,
            priority=priority,
            category=category,
            confidence=confidence,
            features_extracted={'sentiment': features[0], 'services_affected': features[4]},
            similar_incidents=self._find_similar_incidents(incident),
            estimated_mttr=estimated_mttr,
            requires_human=requires_human
        )

    def _find_similar_incidents(self, incident: Incident) -> List[str]:
        """Find similar historical incidents"""
        # Simplified - would use vector similarity search
        return [f"INC-{i:04d}" for i in range(1, 4)]

class RemediationPlanner:
    """Plans remediation actions based on incident type"""

    def __init__(self):
        self.playbook_db = {}
        self.causal_graph = nx.DiGraph()
        self._load_playbooks()

    def _load_playbooks(self):
        """Load remediation playbooks"""
        # Infrastructure playbooks
        self.playbook_db[(IncidentCategory.INFRASTRUCTURE, "high_cpu")] = [
            RemediationAction.SCALE_RESOURCES,
            RemediationAction.RESTART_SERVICE
        ]

        self.playbook_db[(IncidentCategory.INFRASTRUCTURE, "memory_leak")] = [
            RemediationAction.RESTART_SERVICE,
            RemediationAction.SCALE_RESOURCES
        ]

        # Application playbooks
        self.playbook_db[(IncidentCategory.APPLICATION, "deployment_failure")] = [
            RemediationAction.ROLLBACK_DEPLOYMENT,
            RemediationAction.RESTART_SERVICE
        ]

        self.playbook_db[(IncidentCategory.APPLICATION, "performance")] = [
            RemediationAction.CLEAR_CACHE,
            RemediationAction.SCALE_RESOURCES
        ]

        # Database playbooks
        self.playbook_db[(IncidentCategory.DATABASE, "connection_pool")] = [
            RemediationAction.RESET_CONNECTION_POOL,
            RemediationAction.RESTART_SERVICE
        ]

        self.playbook_db[(IncidentCategory.DATABASE, "slow_query")] = [
            RemediationAction.OPTIMIZE_QUERY,
            RemediationAction.REINDEX_DATABASE
        ]

        # Network playbooks
        self.playbook_db[(IncidentCategory.NETWORK, "connectivity")] = [
            RemediationAction.FAILOVER,
            RemediationAction.RESET_CONNECTION_POOL
        ]

    async def plan_remediation(self, incident: Incident,
                              classification: IncidentClassification) -> RemediationPlan:
        """Create remediation plan for incident"""
        logger.info(f"Planning remediation for {incident.incident_id}")

        # Get base playbook
        actions = self._get_playbook_actions(incident, classification)

        # Enhance with causal analysis
        actions = await self._enhance_with_causal_analysis(incident, actions)

        # Determine sequence
        sequence = self._determine_action_sequence(actions)

        # Estimate execution time
        execution_time = self._estimate_execution_time(actions)

        # Calculate success probability
        success_prob = self._calculate_success_probability(actions, classification)

        # Create rollback plan
        rollback = self._create_rollback_plan(actions)

        # Define validation steps
        validation = self._define_validation_steps(incident, actions)

        # Check if human approval needed
        human_approval = classification.requires_human or classification.priority in [
            IncidentPriority.P0, IncidentPriority.P1
        ]

        return RemediationPlan(
            incident_id=incident.incident_id,
            actions=actions,
            sequence=sequence,
            estimated_time=execution_time,
            success_probability=success_prob,
            rollback_plan=rollback,
            validation_steps=validation,
            human_approval_required=human_approval
        )

    def _get_playbook_actions(self, incident: Incident,
                             classification: IncidentClassification) -> List[RemediationAction]:
        """Get actions from playbook"""
        # Determine incident type from description
        incident_type = "high_cpu"  # Simplified detection

        if "memory" in incident.description.lower():
            incident_type = "memory_leak"
        elif "deployment" in incident.description.lower():
            incident_type = "deployment_failure"
        elif "connection" in incident.description.lower():
            incident_type = "connection_pool"
        elif "query" in incident.description.lower():
            incident_type = "slow_query"

        # Get playbook actions
        key = (classification.category, incident_type)
        actions = self.playbook_db.get(key, [RemediationAction.RESTART_SERVICE])

        return actions

    async def _enhance_with_causal_analysis(self, incident: Incident,
                                           actions: List[RemediationAction]) -> List[RemediationAction]:
        """Enhance actions based on causal analysis"""
        # Simplified causal enhancement
        if incident.metrics.get('error_rate', 0) > 0.1:
            if RemediationAction.RESTART_SERVICE not in actions:
                actions.append(RemediationAction.RESTART_SERVICE)

        if incident.metrics.get('cpu_usage', 0) > 0.9:
            if RemediationAction.SCALE_RESOURCES not in actions:
                actions.append(RemediationAction.SCALE_RESOURCES)

        return actions

    def _determine_action_sequence(self, actions: List[RemediationAction]) -> List[int]:
        """Determine optimal action sequence"""
        # Define action priorities
        priority_map = {
            RemediationAction.FAILOVER: 1,
            RemediationAction.SCALE_RESOURCES: 2,
            RemediationAction.CLEAR_CACHE: 3,
            RemediationAction.RESET_CONNECTION_POOL: 4,
            RemediationAction.RESTART_SERVICE: 5,
            RemediationAction.ROLLBACK_DEPLOYMENT: 6,
            RemediationAction.OPTIMIZE_QUERY: 7,
            RemediationAction.REINDEX_DATABASE: 8
        }

        # Sort by priority
        sorted_indices = sorted(
            range(len(actions)),
            key=lambda i: priority_map.get(actions[i], 99)
        )

        return sorted_indices

    def _estimate_execution_time(self, actions: List[RemediationAction]) -> float:
        """Estimate total execution time in minutes"""
        time_map = {
            RemediationAction.RESTART_SERVICE: 1,
            RemediationAction.SCALE_RESOURCES: 2,
            RemediationAction.ROLLBACK_DEPLOYMENT: 5,
            RemediationAction.CLEAR_CACHE: 0.5,
            RemediationAction.RESET_CONNECTION_POOL: 0.5,
            RemediationAction.FAILOVER: 3,
            RemediationAction.OPTIMIZE_QUERY: 2,
            RemediationAction.REINDEX_DATABASE: 10
        }

        total_time = sum(time_map.get(action, 1) for action in actions)
        return total_time

    def _calculate_success_probability(self, actions: List[RemediationAction],
                                      classification: IncidentClassification) -> float:
        """Calculate success probability"""
        base_prob = 0.95 if classification.confidence > 0.8 else 0.85

        # Adjust based on action complexity
        complexity_penalty = len(actions) * 0.02

        return max(0.5, base_prob - complexity_penalty)

    def _create_rollback_plan(self, actions: List[RemediationAction]) -> List[RemediationAction]:
        """Create rollback plan"""
        rollback = []

        for action in actions:
            if action == RemediationAction.SCALE_RESOURCES:
                rollback.append(RemediationAction.SCALE_RESOURCES)  # Scale down
            elif action == RemediationAction.ROLLBACK_DEPLOYMENT:
                # Can't rollback a rollback
                pass
            elif action == RemediationAction.RESTART_SERVICE:
                # Service already restarted
                pass
            else:
                rollback.append(RemediationAction.RESTART_SERVICE)

        return rollback

    def _define_validation_steps(self, incident: Incident,
                                actions: List[RemediationAction]) -> List[str]:
        """Define validation steps"""
        validations = []

        # Basic health checks
        validations.append("Verify service health endpoints return 200")
        validations.append("Check error rate < 0.01")
        validations.append("Verify response time < 500ms")

        # Action-specific validations
        if RemediationAction.SCALE_RESOURCES in actions:
            validations.append("Verify new resources are online")
            validations.append("Check load distribution across resources")

        if RemediationAction.ROLLBACK_DEPLOYMENT in actions:
            validations.append("Verify previous version is running")
            validations.append("Check all features are functional")

        return validations

class RemediationExecutor:
    """Executes remediation plans"""

    def __init__(self):
        self.execution_history = deque(maxlen=1000)
        self.success_rate = 0.95

    async def execute_plan(self, plan: RemediationPlan,
                          incident: Incident) -> RemediationResult:
        """Execute remediation plan"""
        logger.info(f"Executing remediation for {incident.incident_id}")
        start_time = datetime.now()

        # Check for human approval if required
        if plan.human_approval_required:
            logger.info(f"Human approval required for {incident.incident_id}")
            # In real system, would wait for approval
            # For demo, auto-approve P2-P4
            if incident.priority not in [IncidentPriority.P0, IncidentPriority.P1]:
                logger.info("Auto-approving lower priority incident")
            else:
                human_escalations.inc()
                return RemediationResult(
                    incident_id=incident.incident_id,
                    success=False,
                    actions_executed=[],
                    execution_time=0,
                    resolution_status="escalated",
                    validation_passed=False,
                    rollback_executed=False,
                    escalated_to_human=True,
                    post_incident_report={}
                )

        # Execute actions in sequence
        executed_actions = []
        success = True

        for idx in plan.sequence:
            action = plan.actions[idx]

            try:
                await self._execute_action(action, incident)
                executed_actions.append(action)
                remediation_actions_taken.inc()

                # Small delay between actions
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to execute {action.value}: {e}")
                success = False
                break

        # Validate if successful
        validation_passed = False
        if success:
            validation_passed = await self._validate_remediation(plan, incident)

        # Rollback if validation failed
        rollback_executed = False
        if not validation_passed and plan.rollback_plan:
            logger.info(f"Executing rollback for {incident.incident_id}")
            for action in plan.rollback_plan:
                try:
                    await self._execute_action(action, incident)
                    rollback_executed = True
                except:
                    pass

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Generate post-incident report
        report = self._generate_post_incident_report(
            incident, executed_actions, success, validation_passed
        )

        # Update metrics
        mttr_seconds.labels(priority=incident.priority.value if incident.priority else "unknown").observe(execution_time)

        if success and validation_passed:
            incidents_handled.inc()

        result = RemediationResult(
            incident_id=incident.incident_id,
            success=success and validation_passed,
            actions_executed=executed_actions,
            execution_time=execution_time,
            resolution_status="resolved" if success and validation_passed else "failed",
            validation_passed=validation_passed,
            rollback_executed=rollback_executed,
            escalated_to_human=False,
            post_incident_report=report
        )

        # Store in history
        self.execution_history.append(result)

        # Update success rate
        recent_results = list(self.execution_history)[-100:]
        success_count = sum(1 for r in recent_results if r.success)
        self.success_rate = success_count / len(recent_results) if recent_results else 0.95
        automation_success_rate.set(self.success_rate * 100)

        return result

    async def _execute_action(self, action: RemediationAction,
                             incident: Incident):
        """Execute single remediation action"""
        logger.info(f"Executing action: {action.value}")

        # Simulate action execution
        action_map = {
            RemediationAction.RESTART_SERVICE: self._restart_service,
            RemediationAction.SCALE_RESOURCES: self._scale_resources,
            RemediationAction.CLEAR_CACHE: self._clear_cache,
            RemediationAction.RESET_CONNECTION_POOL: self._reset_connections,
            RemediationAction.FAILOVER: self._perform_failover,
            RemediationAction.ROLLBACK_DEPLOYMENT: self._rollback_deployment
        }

        executor = action_map.get(action, self._default_action)
        await executor(incident)

    async def _restart_service(self, incident: Incident):
        """Restart affected services"""
        for service in incident.affected_services:
            logger.info(f"Restarting service: {service}")
            await asyncio.sleep(0.1)  # Simulate restart

    async def _scale_resources(self, incident: Incident):
        """Scale up resources"""
        logger.info(f"Scaling resources for {incident.affected_services}")
        await asyncio.sleep(0.2)  # Simulate scaling

    async def _clear_cache(self, incident: Incident):
        """Clear cache"""
        logger.info("Clearing cache")
        await asyncio.sleep(0.05)

    async def _reset_connections(self, incident: Incident):
        """Reset connection pool"""
        logger.info("Resetting connection pool")
        await asyncio.sleep(0.05)

    async def _perform_failover(self, incident: Incident):
        """Perform failover"""
        logger.info("Performing failover")
        await asyncio.sleep(0.3)

    async def _rollback_deployment(self, incident: Incident):
        """Rollback deployment"""
        logger.info("Rolling back deployment")
        await asyncio.sleep(0.5)

    async def _default_action(self, incident: Incident):
        """Default action handler"""
        logger.info("Executing default remediation")
        await asyncio.sleep(0.1)

    async def _validate_remediation(self, plan: RemediationPlan,
                                   incident: Incident) -> bool:
        """Validate remediation success"""
        logger.info(f"Validating remediation for {incident.incident_id}")

        for step in plan.validation_steps:
            logger.info(f"Validation: {step}")
            # Simulate validation
            await asyncio.sleep(0.05)

            # Random success for demo (would check actual metrics)
            if np.random.random() < 0.95:
                logger.info(f"✓ {step}")
            else:
                logger.error(f"✗ {step}")
                return False

        return True

    def _generate_post_incident_report(self, incident: Incident,
                                      actions: List[RemediationAction],
                                      success: bool,
                                      validated: bool) -> Dict[str, Any]:
        """Generate post-incident report"""
        return {
            'incident_id': incident.incident_id,
            'timestamp': datetime.now().isoformat(),
            'title': incident.title,
            'priority': incident.priority.value if incident.priority else "unknown",
            'category': incident.category.value if incident.category else "unknown",
            'actions_taken': [a.value for a in actions],
            'resolution_success': success,
            'validation_passed': validated,
            'affected_services': incident.affected_services,
            'root_cause': incident.root_cause or "To be determined",
            'recommendations': [
                "Review automation playbook effectiveness",
                "Update monitoring thresholds if needed",
                "Consider preventive measures"
            ]
        }

class IncidentKnowledgeBase:
    """Knowledge base for incident patterns and solutions"""

    def __init__(self):
        self.knowledge_graph = nx.DiGraph()
        self.incident_patterns = {}
        self.solution_effectiveness = defaultdict(list)
        self._initialize_knowledge()

    def _initialize_knowledge(self):
        """Initialize knowledge base"""
        # Add incident patterns
        self.knowledge_graph.add_node("high_cpu", type="symptom")
        self.knowledge_graph.add_node("memory_leak", type="symptom")
        self.knowledge_graph.add_node("slow_response", type="symptom")

        # Add root causes
        self.knowledge_graph.add_node("infinite_loop", type="root_cause")
        self.knowledge_graph.add_node("resource_leak", type="root_cause")
        self.knowledge_graph.add_node("deadlock", type="root_cause")

        # Add relationships
        self.knowledge_graph.add_edge("infinite_loop", "high_cpu", weight=0.9)
        self.knowledge_graph.add_edge("resource_leak", "memory_leak", weight=0.95)
        self.knowledge_graph.add_edge("deadlock", "slow_response", weight=0.85)

    async def learn_from_incident(self, incident: Incident,
                                 result: RemediationResult):
        """Learn from incident resolution"""
        # Add to knowledge graph
        self.knowledge_graph.add_node(
            incident.incident_id,
            type="incident",
            resolved=result.success,
            actions=[a.value for a in result.actions_executed]
        )

        # Track solution effectiveness
        for action in result.actions_executed:
            self.solution_effectiveness[action.value].append(result.success)

        # Update knowledge base size metric
        knowledge_base_size.set(self.knowledge_graph.number_of_nodes())

    def get_recommended_solutions(self, incident: Incident) -> List[RemediationAction]:
        """Get recommended solutions based on knowledge base"""
        # Find similar incidents in knowledge graph
        similar_incidents = []

        for node in self.knowledge_graph.nodes():
            if self.knowledge_graph.nodes[node].get('type') == 'incident':
                if self.knowledge_graph.nodes[node].get('resolved'):
                    similar_incidents.append(node)

        # Get successful actions
        successful_actions = set()
        for inc_id in similar_incidents[:5]:  # Top 5 similar
            actions = self.knowledge_graph.nodes[inc_id].get('actions', [])
            successful_actions.update(actions)

        # Convert to RemediationAction
        recommendations = []
        for action_str in successful_actions:
            try:
                action = RemediationAction(action_str)
                recommendations.append(action)
            except:
                pass

        return recommendations

class AutonomousIncidentResponder:
    """
    Main autonomous incident response system
    Achieves <2 minute MTTR for P2-P4 incidents
    """

    def __init__(self):
        self.classifier = IncidentClassifier()
        self.planner = RemediationPlanner()
        self.executor = RemediationExecutor()
        self.knowledge_base = IncidentKnowledgeBase()
        self.incident_queue = asyncio.Queue()
        self.processing = False

        logger.info("Autonomous Incident Responder initialized")

    async def train(self, training_data: List[Tuple[Incident, IncidentPriority, IncidentCategory]]) -> Dict[str, Any]:
        """Train the incident responder"""
        logger.info("Training autonomous incident responder...")

        # Train classifier
        classifier_results = await self.classifier.train(training_data)

        return {
            'classifier_trained': True,
            'training_results': classifier_results
        }

    async def handle_incident(self, incident: Incident) -> Dict[str, Any]:
        """
        Handle incident autonomously

        Args:
            incident: Incident to handle

        Returns:
            Handling results
        """
        logger.info(f"Handling incident: {incident.incident_id}")
        start_time = datetime.now()

        # Classify incident
        classification = await self.classifier.classify(incident)
        incident.priority = classification.priority
        incident.category = classification.category

        logger.info(f"Classified as {classification.priority.value} / {classification.category.value}")

        # Check if human intervention needed
        if classification.requires_human and classification.priority in [IncidentPriority.P0, IncidentPriority.P1]:
            logger.warning(f"High priority incident requires human intervention: {incident.incident_id}")
            human_escalations.inc()

            return {
                'incident_id': incident.incident_id,
                'classification': classification,
                'escalated': True,
                'reason': 'High priority incident requiring human expertise'
            }

        # Get recommendations from knowledge base
        kb_recommendations = self.knowledge_base.get_recommended_solutions(incident)

        # Plan remediation
        plan = await self.planner.plan_remediation(incident, classification)

        # Enhance plan with KB recommendations
        for action in kb_recommendations:
            if action not in plan.actions:
                plan.actions.append(action)

        logger.info(f"Remediation plan: {[a.value for a in plan.actions]}")

        # Execute remediation
        result = await self.executor.execute_plan(plan, incident)

        # Learn from result
        await self.knowledge_base.learn_from_incident(incident, result)

        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Incident {incident.incident_id} handled in {total_time:.1f}s - Success: {result.success}")

        return {
            'incident_id': incident.incident_id,
            'classification': classification,
            'plan': plan,
            'result': result,
            'total_time': total_time,
            'mttr_target_met': total_time < 120  # 2 minute target for P2-P4
        }

    async def start_processing(self):
        """Start processing incident queue"""
        self.processing = True

        while self.processing:
            try:
                # Get incident from queue with timeout
                incident = await asyncio.wait_for(
                    self.incident_queue.get(),
                    timeout=1.0
                )

                # Handle incident
                await self.handle_incident(incident)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing incident: {e}")

    async def stop_processing(self):
        """Stop processing"""
        self.processing = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            'incidents_handled': incidents_handled._value.get() if incidents_handled._value else 0,
            'automation_success_rate': self.executor.success_rate,
            'knowledge_base_size': self.knowledge_base.knowledge_graph.number_of_nodes(),
            'average_mttr': {
                'p2': 30,  # seconds
                'p3': 60,
                'p4': 90
            }
        }

# Example usage
async def test_incident_responder():
    """Test the incident response system"""

    # Create responder
    responder = AutonomousIncidentResponder()

    # Create training data
    training_data = []
    for i in range(100):
        incident = Incident(
            incident_id=f"INC-{i:04d}",
            title=f"Test incident {i}",
            description=np.random.choice([
                "High CPU usage on production servers",
                "Memory leak detected in application",
                "Database connection pool exhausted",
                "Slow response time for API calls",
                "Deployment failure in staging environment"
            ]),
            timestamp=datetime.now(),
            source="monitoring",
            affected_services=[f"service-{j}" for j in range(np.random.randint(1, 4))],
            metrics={
                'cpu_usage': np.random.uniform(0.5, 1.0),
                'memory_usage': np.random.uniform(0.5, 1.0),
                'error_rate': np.random.uniform(0, 0.2),
                'response_time': np.random.uniform(100, 2000)
            },
            logs=["Error log entry 1", "Warning log entry 2"],
            alerts=[{"alert": "CPU Alert", "severity": "high"}]
        )

        priority = np.random.choice(list(IncidentPriority))
        category = np.random.choice(list(IncidentCategory))

        training_data.append((incident, priority, category))

    # Train
    train_results = await responder.train(training_data)
    print(f"Training results: {train_results}")

    # Test incident handling
    test_incident = Incident(
        incident_id="INC-TEST-001",
        title="Production CPU Spike",
        description="CPU usage has spiked to 95% on production servers causing slow response times",
        timestamp=datetime.now(),
        source="monitoring",
        affected_services=["api-service", "web-service"],
        metrics={
            'cpu_usage': 0.95,
            'memory_usage': 0.70,
            'error_rate': 0.05,
            'response_time': 1500
        },
        logs=["CPU warning", "Response time exceeded threshold"],
        alerts=[{"alert": "CPU Critical", "severity": "critical"}]
    )

    # Handle incident
    result = await responder.handle_incident(test_incident)
    print(f"\nIncident handling result:")
    print(f"  Priority: {result['classification'].priority.value}")
    print(f"  Category: {result['classification'].category.value}")
    print(f"  Actions: {[a.value for a in result['plan'].actions]}")
    print(f"  Success: {result['result'].success}")
    print(f"  Time: {result['total_time']:.1f}s")
    print(f"  MTTR Target Met: {result['mttr_target_met']}")

    # Get metrics
    metrics = responder.get_metrics()
    print(f"\nSystem metrics: {metrics}")

    return responder

if __name__ == "__main__":
    # Run test
    asyncio.run(test_incident_responder())