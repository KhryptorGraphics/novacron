#!/usr/bin/env python3.12
"""
Multi-Cloud VM Orchestrator Example

This example demonstrates advanced features of the Enhanced NovaCron Python SDK:
- Multi-cloud federation management
- AI-powered placement decisions
- Batch operations with progress tracking
- Real-time event streaming
- Cost optimization recommendations
"""

import asyncio
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

from novacron.enhanced_client import (
    EnhancedNovaCronClient,
    CloudProvider,
    AIFeature
)
from novacron.models import CreateVMRequest, VM
from novacron.exceptions import NovaCronException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiCloudOrchestrator:
    """Advanced multi-cloud VM orchestrator with AI capabilities"""
    
    def __init__(self):
        self.client = None
        self.active_clusters = []
        self.cost_thresholds = {
            'hourly': 5.00,    # $5/hour
            'monthly': 3000.00  # $3000/month
        }
    
    async def initialize(self):
        """Initialize the orchestrator with enhanced client"""
        self.client = EnhancedNovaCronClient(
            base_url=os.getenv("NOVACRON_API_URL", "https://api.novacron.io"),
            api_token=os.getenv("NOVACRON_API_TOKEN"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            enable_ai_features=True,
            cloud_provider=CloudProvider.LOCAL,  # Will be overridden per operation
            cache_ttl=300,
            circuit_breaker_threshold=5,
            enable_metrics=True
        )
        
        # Load federated clusters
        await self.discover_clusters()
        
        # Start event monitoring
        asyncio.create_task(self.monitor_federated_events())
        
        logger.info(f"Orchestrator initialized with {len(self.active_clusters)} clusters")
    
    async def discover_clusters(self):
        """Discover all available federated clusters"""
        try:
            clusters = await self.client.list_federated_clusters()
            self.active_clusters = [
                cluster for cluster in clusters 
                if cluster['status'] == 'online'
            ]
            
            logger.info("Available clusters:")
            for cluster in self.active_clusters:
                logger.info(f"  - {cluster['name']}: {cluster['provider']} "
                          f"({cluster['region']}) - {cluster['vm_count']} VMs")
                
        except Exception as e:
            logger.error(f"Failed to discover clusters: {e}")
            self.active_clusters = []
    
    async def deploy_application_tier(
        self,
        app_name: str,
        tier_specs: List[Dict[str, Any]],
        placement_strategy: str = "cost_optimized"
    ) -> Dict[str, List[VM]]:
        """
        Deploy a multi-tier application across optimal cloud providers
        
        Args:
            app_name: Application name
            tier_specs: List of VM specifications for each tier
            placement_strategy: Strategy for placement (cost_optimized, performance, geographic)
        
        Returns:
            Dictionary mapping tier names to created VMs
        """
        logger.info(f"Deploying application '{app_name}' with {len(tier_specs)} tiers")
        
        deployment_plan = await self.create_deployment_plan(
            tier_specs, placement_strategy
        )
        
        results = {}
        
        for tier_name, plan in deployment_plan.items():
            logger.info(f"Deploying tier: {tier_name}")
            
            # Create VM requests
            vm_requests = []
            for i, spec in enumerate(plan['vm_specs']):
                vm_req = CreateVMRequest(
                    name=f"{app_name}-{tier_name}-{i+1}",
                    cpu_shares=spec['cpu_shares'],
                    memory_mb=spec['memory_mb'],
                    disk_size_gb=spec['disk_size_gb'],
                    tags={
                        'application': app_name,
                        'tier': tier_name,
                        'deployment_id': plan['deployment_id'],
                        'strategy': placement_strategy
                    }
                )
                vm_requests.append(vm_req)
            
            # Set client to target cloud provider
            self.client.cloud_provider = CloudProvider(plan['target_provider'])
            self.client.region = plan['target_region']
            
            # Deploy VMs with AI placement
            tier_vms = await self.client.batch_create_vms(
                requests=vm_requests,
                concurrency=min(5, len(vm_requests)),
                use_ai_placement=True
            )
            
            # Filter successful deployments
            successful_vms = [vm for vm in tier_vms if isinstance(vm, VM)]
            failed_deployments = [vm for vm in tier_vms if isinstance(vm, Exception)]
            
            results[tier_name] = successful_vms
            
            logger.info(f"Tier {tier_name}: {len(successful_vms)} VMs deployed, "
                       f"{len(failed_deployments)} failed")
            
            # Log failures
            for i, error in enumerate(failed_deployments):
                logger.error(f"  VM {i+1} failed: {error}")
        
        await self.configure_networking(results)
        return results
    
    async def create_deployment_plan(
        self,
        tier_specs: List[Dict[str, Any]],
        strategy: str
    ) -> Dict[str, Dict[str, Any]]:
        """Create optimal deployment plan using AI recommendations"""
        
        deployment_plan = {}
        
        for i, spec in enumerate(tier_specs):
            tier_name = spec.get('name', f'tier-{i+1}')
            
            if strategy == "cost_optimized":
                plan = await self.plan_cost_optimized_deployment(spec)
            elif strategy == "performance":
                plan = await self.plan_performance_optimized_deployment(spec)
            elif strategy == "geographic":
                plan = await self.plan_geographic_deployment(spec)
            else:
                plan = await self.plan_default_deployment(spec)
            
            deployment_plan[tier_name] = {
                **plan,
                'deployment_id': f"dep-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i}",
                'vm_specs': [spec] * spec.get('replicas', 1)
            }
        
        return deployment_plan
    
    async def plan_cost_optimized_deployment(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Plan deployment optimized for cost"""
        
        # Get cost comparison across providers
        cost_comparisons = []
        
        for cluster in self.active_clusters[:3]:  # Compare top 3 clusters
            try:
                costs = await self.client.get_cross_cloud_costs(
                    source_provider=CloudProvider.LOCAL,
                    target_provider=CloudProvider(cluster['provider']),
                    vm_specs=spec
                )
                
                cost_comparisons.append({
                    'provider': cluster['provider'],
                    'region': cluster['region'],
                    'hourly_cost': costs['hourly_cost'],
                    'monthly_cost': costs['monthly_cost'],
                    'cluster_id': cluster['id']
                })
                
            except Exception as e:
                logger.warning(f"Failed to get costs for {cluster['provider']}: {e}")
        
        # Select most cost-effective option
        if cost_comparisons:
            optimal = min(cost_comparisons, key=lambda x: x['monthly_cost'])
            
            logger.info(f"Cost-optimized placement: {optimal['provider']} "
                       f"(${optimal['monthly_cost']:.2f}/month)")
            
            return {
                'target_provider': optimal['provider'],
                'target_region': optimal['region'],
                'estimated_monthly_cost': optimal['monthly_cost'],
                'placement_reasoning': 'Cost optimized based on provider comparison'
            }
        
        return await self.plan_default_deployment(spec)
    
    async def plan_performance_optimized_deployment(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Plan deployment optimized for performance"""
        
        # Get AI placement recommendation for performance
        recommendation = await self.client.get_intelligent_placement_recommendation(
            vm_specs=spec,
            constraints={'optimization_goal': 'performance'}
        )
        
        # Find cluster with recommended characteristics
        for cluster in self.active_clusters:
            if (cluster['performance_tier'] == 'high' and
                cluster['available_capacity'] > 70):
                
                return {
                    'target_provider': cluster['provider'],
                    'target_region': cluster['region'],
                    'placement_reasoning': recommendation['reasoning'],
                    'expected_performance': 'high'
                }
        
        return await self.plan_default_deployment(spec)
    
    async def plan_geographic_deployment(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Plan deployment based on geographic requirements"""
        
        target_regions = spec.get('target_regions', ['us-west-2'])
        
        # Find clusters in target regions
        regional_clusters = [
            cluster for cluster in self.active_clusters
            if any(region in cluster['region'] for region in target_regions)
        ]
        
        if regional_clusters:
            # Select cluster with best capacity in target region
            optimal = max(regional_clusters, key=lambda x: x['available_capacity'])
            
            return {
                'target_provider': optimal['provider'],
                'target_region': optimal['region'],
                'placement_reasoning': f"Geographic placement in {optimal['region']}",
                'compliance': spec.get('compliance_requirements', [])
            }
        
        return await self.plan_default_deployment(spec)
    
    async def plan_default_deployment(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Default deployment plan"""
        if self.active_clusters:
            default_cluster = self.active_clusters[0]
            return {
                'target_provider': default_cluster['provider'],
                'target_region': default_cluster['region'],
                'placement_reasoning': 'Default cluster selection'
            }
        
        return {
            'target_provider': 'local',
            'target_region': 'default',
            'placement_reasoning': 'No federated clusters available'
        }
    
    async def configure_networking(self, deployment_results: Dict[str, List[VM]]):
        """Configure networking between application tiers"""
        logger.info("Configuring inter-tier networking...")
        
        # This would typically configure:
        # - Load balancers for web tiers
        # - Private networking between tiers
        # - Security groups and firewall rules
        # - Service discovery registration
        
        for tier_name, vms in deployment_results.items():
            logger.info(f"Configuring networking for {tier_name} tier ({len(vms)} VMs)")
            
            # Example: Configure load balancer for web tier
            if 'web' in tier_name.lower():
                await self.setup_load_balancer(tier_name, vms)
            
            # Example: Configure database replication
            if 'db' in tier_name.lower() and len(vms) > 1:
                await self.setup_database_replication(vms)
    
    async def setup_load_balancer(self, tier_name: str, vms: List[VM]):
        """Setup load balancer for web tier VMs"""
        logger.info(f"Setting up load balancer for {tier_name}")
        # Implementation would create cloud provider specific load balancer
        pass
    
    async def setup_database_replication(self, db_vms: List[VM]):
        """Setup database replication between DB VMs"""
        logger.info(f"Setting up database replication for {len(db_vms)} DB VMs")
        # Implementation would configure database replication
        pass
    
    async def monitor_federated_events(self):
        """Monitor events across all federated clusters"""
        logger.info("Starting federated event monitoring")
        
        try:
            # Stream events from all cloud providers
            providers = list(set(
                CloudProvider(cluster['provider']) 
                for cluster in self.active_clusters
            ))
            
            async for event in self.client.stream_federated_events(
                event_types=['vm.created', 'vm.failed', 'vm.migrated', 'cluster.scaled'],
                cloud_providers=providers
            ):
                await self.handle_federated_event(event)
                
        except Exception as e:
            logger.error(f"Event monitoring failed: {e}")
            # Restart monitoring after delay
            await asyncio.sleep(30)
            asyncio.create_task(self.monitor_federated_events())
    
    async def handle_federated_event(self, event: Dict[str, Any]):
        """Handle incoming federated events"""
        event_type = event.get('type')
        event_data = event.get('data', {})
        cloud_provider = event.get('cloud_provider')
        
        logger.info(f"Event from {cloud_provider}: {event_type}")
        
        if event_type == 'vm.failed':
            await self.handle_vm_failure(event_data)
        elif event_type == 'cluster.scaled':
            await self.handle_cluster_scaling(event_data)
        elif event_type == 'vm.migrated':
            await self.handle_vm_migration_complete(event_data)
    
    async def handle_vm_failure(self, event_data: Dict[str, Any]):
        """Handle VM failure events"""
        vm_id = event_data.get('vm_id')
        failure_reason = event_data.get('reason', 'Unknown')
        
        logger.warning(f"VM {vm_id} failed: {failure_reason}")
        
        # Get VM details to determine if it needs replacement
        try:
            vm = await self.client.get_vm(vm_id)
            
            # Check if this was part of an application deployment
            if 'application' in vm.tags:
                app_name = vm.tags['application']
                tier = vm.tags.get('tier', 'unknown')
                
                logger.info(f"Replacing failed VM for {app_name}/{tier}")
                
                # Create replacement VM with same specs
                replacement_request = CreateVMRequest(
                    name=f"{vm.name}-replacement",
                    cpu_shares=vm.config.cpu_shares,
                    memory_mb=vm.config.memory_mb,
                    disk_size_gb=vm.config.disk_size_gb,
                    tags=vm.tags
                )
                
                # Deploy replacement with AI placement
                replacement_vm = await self.client.create_vm_with_ai_placement(
                    replacement_request,
                    use_ai_placement=True,
                    placement_constraints={'exclude_failed_nodes': True}
                )
                
                logger.info(f"Replacement VM created: {replacement_vm.id}")
                
        except Exception as e:
            logger.error(f"Failed to handle VM failure for {vm_id}: {e}")
    
    async def handle_cluster_scaling(self, event_data: Dict[str, Any]):
        """Handle cluster scaling events"""
        cluster_id = event_data.get('cluster_id')
        scale_action = event_data.get('action')  # 'scale_up' or 'scale_down'
        
        logger.info(f"Cluster {cluster_id} scaling: {scale_action}")
        
        # Refresh cluster information
        await self.discover_clusters()
    
    async def handle_vm_migration_complete(self, event_data: Dict[str, Any]):
        """Handle VM migration completion"""
        migration_id = event_data.get('migration_id')
        vm_id = event_data.get('vm_id')
        success = event_data.get('success', False)
        
        if success:
            logger.info(f"Migration {migration_id} completed successfully for VM {vm_id}")
        else:
            logger.warning(f"Migration {migration_id} failed for VM {vm_id}")
    
    async def optimize_costs(self):
        """Analyze and optimize costs across all deployments"""
        logger.info("Analyzing cost optimization opportunities...")
        
        try:
            recommendations = await self.client.get_cost_optimization_recommendations()
            
            total_potential_savings = 0
            
            for rec in recommendations:
                logger.info(f"💰 {rec['title']}")
                logger.info(f"   Current cost: ${rec['current_monthly_cost']:.2f}/month")
                logger.info(f"   Potential savings: ${rec['monthly_savings']:.2f}/month")
                logger.info(f"   Action: {rec['action']}")
                
                total_potential_savings += rec['monthly_savings']
                
                # Auto-implement low-risk optimizations
                if rec['risk_level'] == 'low' and rec['monthly_savings'] > 100:
                    await self.implement_cost_optimization(rec)
            
            logger.info(f"Total potential savings: ${total_potential_savings:.2f}/month")
            
        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {e}")
    
    async def implement_cost_optimization(self, recommendation: Dict[str, Any]):
        """Implement a cost optimization recommendation"""
        action_type = recommendation.get('action_type')
        
        logger.info(f"Implementing cost optimization: {recommendation['title']}")
        
        try:
            if action_type == 'rightsizing':
                await self.implement_rightsizing(recommendation)
            elif action_type == 'provider_migration':
                await self.implement_provider_migration(recommendation)
            elif action_type == 'scheduled_shutdown':
                await self.implement_scheduled_shutdown(recommendation)
            else:
                logger.info(f"Manual action required: {recommendation['action']}")
                
        except Exception as e:
            logger.error(f"Failed to implement optimization: {e}")
    
    async def implement_rightsizing(self, recommendation: Dict[str, Any]):
        """Implement VM rightsizing recommendation"""
        vm_id = recommendation['vm_id']
        new_specs = recommendation['recommended_specs']
        
        # Update VM configuration
        update_request = {
            'cpu_shares': new_specs['cpu_shares'],
            'memory_mb': new_specs['memory_mb']
        }
        
        await self.client.update_vm(vm_id, update_request)
        logger.info(f"Rightsized VM {vm_id}")
    
    async def implement_provider_migration(self, recommendation: Dict[str, Any]):
        """Implement cross-cloud provider migration"""
        vm_id = recommendation['vm_id']
        target_provider = recommendation['target_provider']
        target_cluster = recommendation['target_cluster']
        
        # Create cross-cloud migration
        migration = await self.client.create_cross_cloud_migration(
            vm_id=vm_id,
            target_cluster=target_cluster,
            target_provider=CloudProvider(target_provider),
            target_region=recommendation['target_region']
        )
        
        logger.info(f"Started cross-cloud migration {migration.id} for VM {vm_id}")
    
    async def implement_scheduled_shutdown(self, recommendation: Dict[str, Any]):
        """Implement scheduled VM shutdown for dev/test environments"""
        vm_ids = recommendation['vm_ids']
        schedule = recommendation['schedule']  # e.g., "weekends", "nights"
        
        # This would typically integrate with a scheduler service
        logger.info(f"Scheduled shutdown configured for {len(vm_ids)} VMs: {schedule}")
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestrator report"""
        
        # Get performance metrics
        request_metrics = self.client.get_request_metrics()
        circuit_breaker_status = self.client.get_circuit_breaker_status()
        
        # Analyze deployments
        active_deployments = len([
            cluster for cluster in self.active_clusters 
            if cluster['vm_count'] > 0
        ])
        
        total_vms = sum(cluster['vm_count'] for cluster in self.active_clusters)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'clusters': {
                'total': len(self.active_clusters),
                'active_deployments': active_deployments,
                'total_vms': total_vms
            },
            'performance': {
                'api_requests': len(request_metrics),
                'avg_response_time': sum(
                    m['avg_duration'] for m in request_metrics.values()
                ) / len(request_metrics) if request_metrics else 0,
                'circuit_breakers_open': sum(
                    1 for status in circuit_breaker_status.values() 
                    if status.get('is_open', False)
                )
            },
            'providers': list(set(
                cluster['provider'] for cluster in self.active_clusters
            ))
        }
        
        return report
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.close()
        logger.info("Orchestrator cleanup completed")


async def main():
    """Main orchestrator demonstration"""
    
    # Example application tiers specification
    web_app_tiers = [
        {
            'name': 'web',
            'cpu_shares': 2048,
            'memory_mb': 4096,
            'disk_size_gb': 50,
            'replicas': 3,
            'target_regions': ['us-west-2', 'us-east-1']
        },
        {
            'name': 'api',
            'cpu_shares': 4096,
            'memory_mb': 8192,
            'disk_size_gb': 100,
            'replicas': 2,
            'target_regions': ['us-west-2']
        },
        {
            'name': 'database',
            'cpu_shares': 8192,
            'memory_mb': 16384,
            'disk_size_gb': 500,
            'replicas': 2,
            'target_regions': ['us-west-2'],
            'compliance_requirements': ['PCI-DSS']
        }
    ]
    
    orchestrator = MultiCloudOrchestrator()
    
    try:
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Deploy application with cost optimization
        logger.info("Deploying e-commerce application...")
        deployment_results = await orchestrator.deploy_application_tier(
            app_name="ecommerce-prod",
            tier_specs=web_app_tiers,
            placement_strategy="cost_optimized"
        )
        
        # Report deployment results
        total_vms = sum(len(vms) for vms in deployment_results.values())
        logger.info(f"Deployment completed: {total_vms} VMs across {len(deployment_results)} tiers")
        
        for tier, vms in deployment_results.items():
            logger.info(f"  {tier}: {len(vms)} VMs")
        
        # Run cost optimization analysis
        await orchestrator.optimize_costs()
        
        # Generate and display report
        report = await orchestrator.generate_report()
        logger.info("=== Orchestrator Report ===")
        logger.info(f"Total clusters: {report['clusters']['total']}")
        logger.info(f"Active deployments: {report['clusters']['active_deployments']}")
        logger.info(f"Total VMs: {report['clusters']['total_vms']}")
        logger.info(f"Avg response time: {report['performance']['avg_response_time']:.3f}s")
        logger.info(f"Cloud providers: {', '.join(report['providers'])}")
        
        # Keep monitoring for a while
        logger.info("Monitoring events for 60 seconds...")
        await asyncio.sleep(60)
        
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())