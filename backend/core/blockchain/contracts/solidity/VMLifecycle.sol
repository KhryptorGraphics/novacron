// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title VMLifecycle
 * @dev Smart contract for VM lifecycle management
 */
contract VMLifecycle {

    enum VMState { Stopped, Starting, Running, Migrating, Paused, Destroyed }

    struct VM {
        uint256 vmId;
        address owner;
        string region;
        VMState state;
        uint256 cpuAllocation;
        uint256 memoryAllocation;
        uint256 storageAllocation;
        uint256 networkBandwidth;
        uint256 createdAt;
        uint256 lastModified;
        string ipfsHash;
    }

    mapping(uint256 => VM) public vms;
    mapping(address => uint256[]) public ownerVMs;
    uint256 public vmCount;

    // Multi-signature settings
    mapping(uint256 => mapping(address => bool)) public migrationApprovals;
    mapping(uint256 => uint256) public migrationApprovalCount;
    uint256 public requiredApprovals = 2;

    event VMCreated(uint256 indexed vmId, address indexed owner, string region);
    event VMStateChanged(uint256 indexed vmId, VMState oldState, VMState newState);
    event VMMigrationInitiated(uint256 indexed vmId, string targetRegion);
    event VMMigrationApproved(uint256 indexed vmId, address indexed approver);
    event VMMigrationCompleted(uint256 indexed vmId, string newRegion);
    event VMDestroyed(uint256 indexed vmId);

    modifier onlyOwner(uint256 vmId) {
        require(vms[vmId].owner == msg.sender, "Not VM owner");
        _;
    }

    modifier vmExists(uint256 vmId) {
        require(vms[vmId].owner != address(0), "VM does not exist");
        _;
    }

    /**
     * @dev Create a new VM
     */
    function createVM(
        string memory region,
        uint256 cpuAllocation,
        uint256 memoryAllocation,
        uint256 storageAllocation,
        uint256 networkBandwidth,
        string memory ipfsHash
    ) public returns (uint256) {
        vmCount++;
        uint256 vmId = vmCount;

        vms[vmId] = VM({
            vmId: vmId,
            owner: msg.sender,
            region: region,
            state: VMState.Stopped,
            cpuAllocation: cpuAllocation,
            memoryAllocation: memoryAllocation,
            storageAllocation: storageAllocation,
            networkBandwidth: networkBandwidth,
            createdAt: block.timestamp,
            lastModified: block.timestamp,
            ipfsHash: ipfsHash
        });

        ownerVMs[msg.sender].push(vmId);

        emit VMCreated(vmId, msg.sender, region);

        return vmId;
    }

    /**
     * @dev Start a VM
     */
    function startVM(uint256 vmId) public onlyOwner(vmId) vmExists(vmId) {
        require(vms[vmId].state == VMState.Stopped, "VM must be stopped");

        VMState oldState = vms[vmId].state;
        vms[vmId].state = VMState.Starting;
        vms[vmId].lastModified = block.timestamp;

        emit VMStateChanged(vmId, oldState, VMState.Starting);
    }

    /**
     * @dev Stop a VM
     */
    function stopVM(uint256 vmId) public onlyOwner(vmId) vmExists(vmId) {
        require(vms[vmId].state == VMState.Running, "VM must be running");

        VMState oldState = vms[vmId].state;
        vms[vmId].state = VMState.Stopped;
        vms[vmId].lastModified = block.timestamp;

        emit VMStateChanged(vmId, oldState, VMState.Stopped);
    }

    /**
     * @dev Initiate VM migration with multi-sig
     */
    function initiateMigration(uint256 vmId, string memory targetRegion)
        public
        onlyOwner(vmId)
        vmExists(vmId)
    {
        require(vms[vmId].state == VMState.Running, "VM must be running");

        emit VMMigrationInitiated(vmId, targetRegion);
    }

    /**
     * @dev Approve VM migration (multi-sig)
     */
    function approveMigration(uint256 vmId) public vmExists(vmId) {
        require(!migrationApprovals[vmId][msg.sender], "Already approved");

        migrationApprovals[vmId][msg.sender] = true;
        migrationApprovalCount[vmId]++;

        emit VMMigrationApproved(vmId, msg.sender);
    }

    /**
     * @dev Complete VM migration after approvals
     */
    function completeMigration(uint256 vmId, string memory newRegion)
        public
        onlyOwner(vmId)
        vmExists(vmId)
    {
        require(
            migrationApprovalCount[vmId] >= requiredApprovals,
            "Insufficient approvals"
        );

        vms[vmId].region = newRegion;
        vms[vmId].state = VMState.Running;
        vms[vmId].lastModified = block.timestamp;

        // Reset migration approvals
        migrationApprovalCount[vmId] = 0;

        emit VMMigrationCompleted(vmId, newRegion);
    }

    /**
     * @dev Update VM state
     */
    function updateVMState(uint256 vmId, VMState newState)
        public
        onlyOwner(vmId)
        vmExists(vmId)
    {
        VMState oldState = vms[vmId].state;
        vms[vmId].state = newState;
        vms[vmId].lastModified = block.timestamp;

        emit VMStateChanged(vmId, oldState, newState);
    }

    /**
     * @dev Update IPFS hash
     */
    function updateIPFSHash(uint256 vmId, string memory ipfsHash)
        public
        onlyOwner(vmId)
        vmExists(vmId)
    {
        vms[vmId].ipfsHash = ipfsHash;
        vms[vmId].lastModified = block.timestamp;
    }

    /**
     * @dev Destroy a VM
     */
    function destroyVM(uint256 vmId) public onlyOwner(vmId) vmExists(vmId) {
        vms[vmId].state = VMState.Destroyed;
        vms[vmId].lastModified = block.timestamp;

        emit VMDestroyed(vmId);
    }

    /**
     * @dev Get VM details
     */
    function getVM(uint256 vmId) public view returns (VM memory) {
        return vms[vmId];
    }

    /**
     * @dev Get VMs owned by address
     */
    function getOwnerVMs(address owner) public view returns (uint256[] memory) {
        return ownerVMs[owner];
    }
}
