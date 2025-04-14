package network

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// createBridgeNetwork creates a bridge network
func (m *NetworkManager) createBridgeNetwork(network *Network) error {
	// Check if we can use the docker bridge driver as a helper
	if isDockerAvailable() {
		return createDockerNetwork(network, "bridge")
	}

	// Fall back to manual bridge creation
	bridgeName := fmt.Sprintf("br-%s", network.ID[:8])
	
	// Create the bridge interface
	cmd := exec.Command("ip", "link", "add", "name", bridgeName, "type", "bridge")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create bridge interface: %w", err)
	}
	
	// Set the bridge up
	cmd = exec.Command("ip", "link", "set", bridgeName, "up")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to set bridge up: %w", err)
	}
	
	// Configure IP addressing
	_, ipNet, err := net.ParseCIDR(network.IPAM.Subnet)
	if err != nil {
		return fmt.Errorf("failed to parse subnet: %w", err)
	}
	
	// Assign the gateway IP to the bridge
	cmd = exec.Command("ip", "addr", "add", network.IPAM.Gateway+"/"+strings.Split(network.IPAM.Subnet, "/")[1], "dev", bridgeName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to assign IP to bridge: %w", err)
	}
	
	// Store the bridge name in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["bridge_name"] = bridgeName
	
	// Save network configuration to disk
	if err := saveNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to save network configuration: %v", err)
	}
	
	log.Printf("Created bridge network %s with bridge %s", network.Name, bridgeName)
	return nil
}

// createOverlayNetwork creates an overlay network
func (m *NetworkManager) createOverlayNetwork(network *Network) error {
	// Check if we can use docker swarm as a helper
	if isDockerSwarmAvailable() {
		return createDockerNetwork(network, "overlay")
	}
	
	// Fall back to manual overlay creation
	// This is a simplified example - in a real system, you'd:
	// 1. Set up VXLAN tunnels
	// 2. Configure routing
	// 3. Set up distributed key-value store for coordination
	
	log.Printf("Creating overlay network %s (simplified implementation)", network.Name)
	
	// Store a placeholder in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["overlay_id"] = fmt.Sprintf("vxlan-%s", network.ID[:8])
	
	// Save network configuration to disk
	if err := saveNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to save network configuration: %v", err)
	}
	
	return nil
}

// createMacvlanNetwork creates a macvlan network
func (m *NetworkManager) createMacvlanNetwork(network *Network) error {
	// Check if we can use docker macvlan as a helper
	if isDockerAvailable() {
		return createDockerNetwork(network, "macvlan")
	}
	
	// Fall back to manual macvlan creation
	// Determine the parent interface
	parentIface := network.Options["parent"]
	if parentIface == "" {
		// Try to find a default interface
		iface, err := getDefaultInterface()
		if err != nil {
			return fmt.Errorf("failed to determine parent interface: %w", err)
		}
		parentIface = iface
	}
	
	// Create the macvlan interface
	macvlanName := fmt.Sprintf("mcv-%s", network.ID[:8])
	cmd := exec.Command("ip", "link", "add", macvlanName, "link", parentIface, "type", "macvlan", "mode", "bridge")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create macvlan interface: %w", err)
	}
	
	// Set the macvlan up
	cmd = exec.Command("ip", "link", "set", macvlanName, "up")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to set macvlan up: %w", err)
	}
	
	// Store the macvlan name in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["macvlan_name"] = macvlanName
	network.Options["parent"] = parentIface
	
	// Save network configuration to disk
	if err := saveNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to save network configuration: %v", err)
	}
	
	log.Printf("Created macvlan network %s with interface %s on parent %s", network.Name, macvlanName, parentIface)
	return nil
}

// Network deletion implementations for different network types

// deleteBridgeNetwork deletes a bridge network
func (m *NetworkManager) deleteBridgeNetwork(network *Network) error {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// Fall back to manual bridge deletion
	bridgeName := network.Options["bridge_name"]
	if bridgeName == "" {
		bridgeName = fmt.Sprintf("br-%s", network.ID[:8])
	}
	
	// Set the bridge down
	cmd := exec.Command("ip", "link", "set", bridgeName, "down")
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: Failed to set bridge down: %v", err)
	}
	
	// Delete the bridge interface
	cmd = exec.Command("ip", "link", "del", bridgeName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete bridge interface: %w", err)
	}
	
	// Delete network configuration file
	if err := deleteNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to delete network configuration: %v", err)
	}
	
	log.Printf("Deleted bridge network %s with bridge %s", network.Name, bridgeName)
	return nil
}

// deleteOverlayNetwork deletes an overlay network
func (m *NetworkManager) deleteOverlayNetwork(network *Network) error {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// Fall back to manual overlay deletion
	log.Printf("Deleting overlay network %s (simplified implementation)", network.Name)
	
	// Delete network configuration file
	if err := deleteNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to delete network configuration: %v", err)
	}
	
	return nil
}

// deleteMacvlanNetwork deletes a macvlan network
func (m *NetworkManager) deleteMacvlanNetwork(network *Network) error {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		return deleteDockerNetwork(network.Options["docker_network_id"])
	}
	
	// Fall back to manual macvlan deletion
	macvlanName := network.Options["macvlan_name"]
	if macvlanName == "" {
		macvlanName = fmt.Sprintf("mcv-%s", network.ID[:8])
	}
	
	// Set the macvlan down
	cmd := exec.Command("ip", "link", "set", macvlanName, "down")
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: Failed to set macvlan down: %v", err)
	}
	
	// Delete the macvlan interface
	cmd = exec.Command("ip", "link", "del", macvlanName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to delete macvlan interface: %w", err)
	}
	
	// Delete network configuration file
	if err := deleteNetworkConfig(network); err != nil {
		log.Printf("Warning: Failed to delete network configuration: %v", err)
	}
	
	log.Printf("Deleted macvlan network %s with interface %s", network.Name, macvlanName)
	return nil
}

// Helper functions

// isDockerAvailable checks if Docker is available
func isDockerAvailable() bool {
	cmd := exec.Command("docker", "version")
	return cmd.Run() == nil
}

// isDockerSwarmAvailable checks if Docker Swarm is available
func isDockerSwarmAvailable() bool {
	cmd := exec.Command("docker", "info", "--format", "{{.Swarm.LocalNodeState}}")
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	
	return strings.TrimSpace(string(output)) == "active"
}

// createDockerNetwork creates a network using Docker
func createDockerNetwork(network *Network, driver string) error {
	args := []string{
		"network", "create",
		"--driver", driver,
		"--subnet", network.IPAM.Subnet,
	}
	
	if network.IPAM.Gateway != "" {
		args = append(args, "--gateway", network.IPAM.Gateway)
	}
	
	if network.IPAM.IPRange != "" {
		args = append(args, "--ip-range", network.IPAM.IPRange)
	}
	
	if network.Internal {
		args = append(args, "--internal")
	}
	
	if network.EnableIPv6 {
		args = append(args, "--ipv6")
	}
	
	// Add labels
	for k, v := range network.Labels {
		args = append(args, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Add options
	for k, v := range network.Options {
		args = append(args, "--opt", fmt.Sprintf("%s=%s", k, v))
	}
	
	// Add the network name
	args = append(args, network.Name)
	
	// Create the network
	cmd := exec.Command("docker", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create docker network: %w, output: %s", err, string(output))
	}
	
	// Get the network ID
	networkID := strings.TrimSpace(string(output))
	
	// Store the docker network ID in the network options
	if network.Options == nil {
		network.Options = make(map[string]string)
	}
	network.Options["docker_network_id"] = networkID
	
	log.Printf("Created docker %s network %s with ID %s", driver, network.Name, networkID)
	return nil
}

// deleteDockerNetwork deletes a network using Docker
func deleteDockerNetwork(networkID string) error {
	cmd := exec.Command("docker", "network", "rm", networkID)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to delete docker network: %w, output: %s", err, string(output))
	}
	
	log.Printf("Deleted docker network with ID %s", networkID)
	return nil
}

// getDefaultInterface returns the default network interface
func getDefaultInterface() (string, error) {
	// Get the default route
	cmd := exec.Command("ip", "route", "show", "default")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get default route: %w", err)
	}
	
	// Parse the output to get the interface
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "default") {
			fields := strings.Fields(line)
			for i, field := range fields {
				if field == "dev" && i+1 < len(fields) {
					return fields[i+1], nil
				}
			}
		}
	}
	
	return "", fmt.Errorf("failed to determine default interface")
}

// saveNetworkConfig saves network configuration to disk
func saveNetworkConfig(network *Network) error {
	// Create the networks directory if it doesn't exist
	networksDir := "/var/lib/novacron/networks"
	if err := os.MkdirAll(networksDir, 0755); err != nil {
		return fmt.Errorf("failed to create networks directory: %w", err)
	}
	
	// Create the network configuration file
	configPath := filepath.Join(networksDir, network.ID+".json")
	configData, err := json.MarshalIndent(network, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal network configuration: %w", err)
	}
	
	if err := os.WriteFile(configPath, configData, 0644); err != nil {
		return fmt.Errorf("failed to write network configuration: %w", err)
	}
	
	return nil
}

// deleteNetworkConfig deletes network configuration from disk
func deleteNetworkConfig(network *Network) error {
	configPath := filepath.Join("/var/lib/novacron/networks", network.ID+".json")
	if err := os.Remove(configPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete network configuration: %w", err)
	}
	
	return nil
}

// loadNetworks loads existing networks from disk
func (m *NetworkManager) loadNetworks() error {
	networksDir := "/var/lib/novacron/networks"
	if _, err := os.Stat(networksDir); os.IsNotExist(err) {
		// Directory doesn't exist, no networks to load
		return nil
	}
	
	// Read the networks directory
	files, err := os.ReadDir(networksDir)
	if err != nil {
		return fmt.Errorf("failed to read networks directory: %w", err)
	}
	
	for _, file := range files {
		// Skip directories and non-JSON files
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}
		
		// Read the network configuration
		configPath := filepath.Join(networksDir, file.Name())
		configData, err := os.ReadFile(configPath)
		if err != nil {
			log.Printf("Warning: Failed to read network configuration %s: %v", configPath, err)
			continue
		}
		
		// Parse the network configuration
		var network Network
		if err := json.Unmarshal(configData, &network); err != nil {
			log.Printf("Warning: Failed to parse network configuration %s: %v", configPath, err)
			continue
		}
		
		// Add the network to our maps
		m.networksMutex.Lock()
		m.networks[network.ID] = &network
		m.networksByName[network.Name] = network.ID
		m.networksMutex.Unlock()
		
		log.Printf("Loaded network %s (ID: %s) from disk", network.Name, network.ID)
	}
	
	return nil
}

// updateNetworks periodically updates network information
func (m *NetworkManager) updateNetworks() {
	ticker := time.NewTicker(m.config.UpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.updateNetworkInfo()
		}
	}
}

// updateNetworkInfo updates information about all networks
func (m *NetworkManager) updateNetworkInfo() {
	m.networksMutex.Lock()
	defer m.networksMutex.Unlock()
	
	for _, network := range m.networks {
		// Update network information based on type
		switch network.Type {
		case NetworkTypeBridge:
			m.updateBridgeNetworkInfo(network)
		case NetworkTypeOverlay:
			m.updateOverlayNetworkInfo(network)
		case NetworkTypeMacvlan:
			m.updateMacvlanNetworkInfo(network)
		}
		
		network.NetworkInfo.LastUpdated = time.Now()
	}
}

// updateBridgeNetworkInfo updates information about a bridge network
func (m *NetworkManager) updateBridgeNetworkInfo(network *Network) {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		// Update using docker
		cmd := exec.Command("docker", "network", "inspect", network.Options["docker_network_id"])
		output, err := cmd.Output()
		if err != nil {
			log.Printf("Warning: Failed to inspect docker network %s: %v", network.Options["docker_network_id"], err)
			network.NetworkInfo.Active = false
			return
		}
		
		// Network exists, mark as active
		network.NetworkInfo.Active = true
		return
	}
	
	// Fall back to manual bridge inspection
	bridgeName := network.Options["bridge_name"]
	if bridgeName == "" {
		bridgeName = fmt.Sprintf("br-%s", network.ID[:8])
	}
	
	// Check if the bridge exists
	cmd := exec.Command("ip", "link", "show", bridgeName)
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: Bridge %s not found: %v", bridgeName, err)
		network.NetworkInfo.Active = false
		return
	}
	
	// Bridge exists, mark as active
	network.NetworkInfo.Active = true
}

// updateOverlayNetworkInfo updates information about an overlay network
func (m *NetworkManager) updateOverlayNetworkInfo(network *Network) {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		// Update using docker
		cmd := exec.Command("docker", "network", "inspect", network.Options["docker_network_id"])
		output, err := cmd.Output()
		if err != nil {
			log.Printf("Warning: Failed to inspect docker network %s: %v", network.Options["docker_network_id"], err)
			network.NetworkInfo.Active = false
			return
		}
		
		// Network exists, mark as active
		network.NetworkInfo.Active = true
		return
	}
	
	// Fall back to manual overlay inspection
	// This is a simplified implementation
	network.NetworkInfo.Active = true
}

// updateMacvlanNetworkInfo updates information about a macvlan network
func (m *NetworkManager) updateMacvlanNetworkInfo(network *Network) {
	// Check if we used docker to create the network
	if network.Options != nil && network.Options["docker_network_id"] != "" {
		// Update using docker
		cmd := exec.Command("docker", "network", "inspect", network.Options["docker_network_id"])
		output, err := cmd.Output()
		if err != nil {
			log.Printf("Warning: Failed to inspect docker network %s: %v", network.Options["docker_network_id"], err)
			network.NetworkInfo.Active = false
			return
		}
		
		// Network exists, mark as active
		network.NetworkInfo.Active = true
		return
	}
	
	// Fall back to manual macvlan inspection
	macvlanName := network.Options["macvlan_name"]
	if macvlanName == "" {
		macvlanName = fmt.Sprintf("mcv-%s", network.ID[:8])
	}
	
	// Check if the macvlan exists
	cmd := exec.Command("ip", "link", "show", macvlanName)
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: Macvlan %s not found: %v", macvlanName, err)
		network.NetworkInfo.Active = false
		return
	}
	
	// Macvlan exists, mark as active
	network.NetworkInfo.Active = true
}

// emitEvent emits a network event to all listeners
func (m *NetworkManager) emitEvent(event NetworkEvent) {
	m.eventMutex.RLock()
	defer m.eventMutex.RUnlock()
	
	for _, listener := range m.eventListeners {
		go listener(event)
	}
}
