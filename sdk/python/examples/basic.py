"""
Basic DWCP Python SDK Example

Demonstrates VM creation, management, and monitoring.
"""

import asyncio
from dwcp import Client, ClientConfig, VMConfig, NetworkConfig, NetworkInterface


async def main():
    # Create client configuration
    config = ClientConfig(
        address="localhost",
        port=9000,
        api_key="your-api-key",
        tls_enabled=True,
    )

    # Use async context manager for automatic cleanup
    async with Client(config) as client:
        print("Connected to DWCP server")

        # Create VM configuration
        vm_config = VMConfig(
            name="example-vm",
            memory=2 * 1024 ** 3,  # 2GB
            cpus=2,
            disk=20 * 1024 ** 3,  # 20GB
            image="ubuntu-22.04",
            network=NetworkConfig(
                mode="bridge",
                interfaces=[
                    NetworkInterface(
                        name="eth0",
                        type="virtio",
                    ),
                ],
            ),
            labels={
                "env": "production",
                "team": "platform",
            },
        )

        # Get VM client
        vm_client = client.VM()

        # Create VM
        vm = await vm_client.create(vm_config)
        print(f"Created VM: {vm.name} (ID: {vm.id})")

        # Start VM
        await vm_client.start(vm.id)
        print("VM started successfully")

        # Watch VM events for 30 seconds
        print("Monitoring VM events...")
        timeout = asyncio.create_task(asyncio.sleep(30))

        async for event in vm_client.watch(vm.id):
            if timeout.done():
                break

            print(f"Event: {event.type} - {event.message} (State: {event.vm.state})")

        print("Monitoring complete")

        # Get VM metrics
        metrics = await vm_client.get_metrics(vm.id)
        print(f"\nVM Metrics:")
        print(f"  CPU Usage: {metrics.cpu_usage:.2f}%")
        print(f"  Memory Used: {metrics.memory_used / 1024 ** 3:.2f} GB")
        print(f"  Network RX: {metrics.network_rx / 1024 ** 2:.2f} MB")
        print(f"  Network TX: {metrics.network_tx / 1024 ** 2:.2f} MB")

        # Cleanup
        print("\nCleaning up...")
        await vm_client.stop(vm.id)
        await vm_client.destroy(vm.id)
        print("Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
