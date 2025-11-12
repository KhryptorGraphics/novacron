// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title NovaCron Decentralized VM Registry
 * @notice Smart contracts for blockchain-based VM lifecycle management
 * @dev Implements decentralized registry, resource tokenization, and consensus mechanisms
 */

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

// VM Registry Contract
contract VMRegistry is AccessControl, ReentrancyGuard {
    using EnumerableSet for EnumerableSet.Bytes32Set;
    using ECDSA for bytes32;

    // Roles
    bytes32 public constant PROVIDER_ROLE = keccak256("PROVIDER_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");

    // VM States
    enum VMState {
        Idle,
        Provisioning,
        Running,
        Migrating,
        Suspended,
        Terminated
    }

    // VM Configuration
    struct VMConfig {
        uint256 cpuCores;
        uint256 memoryGB;
        uint256 storageGB;
        uint256 networkBandwidthMbps;
        bool gpuEnabled;
        string osType;
        string region;
    }

    // VM Metadata
    struct VMMetadata {
        bytes32 id;
        address owner;
        address provider;
        VMConfig config;
        VMState state;
        uint256 createdAt;
        uint256 lastUpdated;
        uint256 uptime;
        bytes32 contentHash; // IPFS hash of VM image
        uint256 tokenId; // NFT token ID
    }

    // Provider Information
    struct Provider {
        address addr;
        string name;
        string endpoint;
        uint256 totalCapacity;
        uint256 usedCapacity;
        uint256 reputation;
        uint256 stakedAmount;
        bool active;
    }

    // Service Level Agreement
    struct SLA {
        uint256 uptimeGuarantee; // in basis points (9999 = 99.99%)
        uint256 responseTime; // in milliseconds
        uint256 throughput; // requests per second
        uint256 penaltyRate; // penalty per violation
        uint256 violationCount;
    }

    // Storage
    mapping(bytes32 => VMMetadata) public vms;
    mapping(address => Provider) public providers;
    mapping(bytes32 => SLA) public slas;
    mapping(address => EnumerableSet.Bytes32Set) private userVMs;
    mapping(address => EnumerableSet.Bytes32Set) private providerVMs;

    EnumerableSet.Bytes32Set private allVMs;

    // Events
    event VMCreated(bytes32 indexed vmId, address indexed owner, address indexed provider);
    event VMStateChanged(bytes32 indexed vmId, VMState oldState, VMState newState);
    event VMMigrated(bytes32 indexed vmId, address oldProvider, address newProvider);
    event ProviderRegistered(address indexed provider, string name);
    event SLAViolation(bytes32 indexed vmId, address indexed provider, uint256 penalty);

    // Modifiers
    modifier onlyVMOwner(bytes32 vmId) {
        require(vms[vmId].owner == msg.sender, "Not VM owner");
        _;
    }

    modifier onlyProvider() {
        require(hasRole(PROVIDER_ROLE, msg.sender), "Not a provider");
        _;
    }

    modifier vmExists(bytes32 vmId) {
        require(vms[vmId].createdAt > 0, "VM does not exist");
        _;
    }

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(GOVERNANCE_ROLE, msg.sender);
    }

    /**
     * @notice Register as a compute provider
     * @param name Provider name
     * @param endpoint API endpoint
     * @param capacity Total compute capacity
     */
    function registerProvider(
        string memory name,
        string memory endpoint,
        uint256 capacity
    ) external payable {
        require(msg.value >= 10 ether, "Insufficient stake");
        require(bytes(name).length > 0, "Invalid name");

        providers[msg.sender] = Provider({
            addr: msg.sender,
            name: name,
            endpoint: endpoint,
            totalCapacity: capacity,
            usedCapacity: 0,
            reputation: 1000, // Starting reputation
            stakedAmount: msg.value,
            active: true
        });

        _setupRole(PROVIDER_ROLE, msg.sender);
        emit ProviderRegistered(msg.sender, name);
    }

    /**
     * @notice Create a new VM
     * @param config VM configuration
     * @param provider Selected provider address
     * @return vmId Unique VM identifier
     */
    function createVM(
        VMConfig memory config,
        address provider
    ) external payable returns (bytes32 vmId) {
        require(providers[provider].active, "Provider not active");
        require(msg.value >= calculateVMCost(config), "Insufficient payment");

        vmId = keccak256(abi.encodePacked(msg.sender, provider, block.timestamp));

        vms[vmId] = VMMetadata({
            id: vmId,
            owner: msg.sender,
            provider: provider,
            config: config,
            state: VMState.Provisioning,
            createdAt: block.timestamp,
            lastUpdated: block.timestamp,
            uptime: 0,
            contentHash: bytes32(0),
            tokenId: 0
        });

        allVMs.add(vmId);
        userVMs[msg.sender].add(vmId);
        providerVMs[provider].add(vmId);

        // Update provider capacity
        uint256 requiredCapacity = calculateRequiredCapacity(config);
        providers[provider].usedCapacity += requiredCapacity;

        emit VMCreated(vmId, msg.sender, provider);

        return vmId;
    }

    /**
     * @notice Update VM state
     * @param vmId VM identifier
     * @param newState New VM state
     */
    function updateVMState(bytes32 vmId, VMState newState)
        external
        vmExists(vmId)
        onlyProvider
    {
        require(vms[vmId].provider == msg.sender, "Not VM provider");

        VMState oldState = vms[vmId].state;
        vms[vmId].state = newState;
        vms[vmId].lastUpdated = block.timestamp;

        if (newState == VMState.Running && oldState != VMState.Running) {
            vms[vmId].uptime = block.timestamp;
        }

        emit VMStateChanged(vmId, oldState, newState);
    }

    /**
     * @notice Migrate VM to new provider
     * @param vmId VM identifier
     * @param newProvider New provider address
     */
    function migrateVM(bytes32 vmId, address newProvider)
        external
        vmExists(vmId)
        onlyVMOwner(vmId)
    {
        require(providers[newProvider].active, "New provider not active");

        address oldProvider = vms[vmId].provider;

        // Update capacity
        uint256 requiredCapacity = calculateRequiredCapacity(vms[vmId].config);
        providers[oldProvider].usedCapacity -= requiredCapacity;
        providers[newProvider].usedCapacity += requiredCapacity;

        // Update VM metadata
        vms[vmId].provider = newProvider;
        vms[vmId].state = VMState.Migrating;
        vms[vmId].lastUpdated = block.timestamp;

        // Update provider mappings
        providerVMs[oldProvider].remove(vmId);
        providerVMs[newProvider].add(vmId);

        emit VMMigrated(vmId, oldProvider, newProvider);
    }

    /**
     * @notice Calculate VM cost based on configuration
     * @param config VM configuration
     * @return cost Cost in wei
     */
    function calculateVMCost(VMConfig memory config) public pure returns (uint256) {
        uint256 baseCost = 0.001 ether;
        uint256 cpuCost = config.cpuCores * 0.0001 ether;
        uint256 memoryCost = config.memoryGB * 0.00005 ether;
        uint256 storageCost = config.storageGB * 0.00001 ether;
        uint256 networkCost = config.networkBandwidthMbps * 0.000001 ether;
        uint256 gpuCost = config.gpuEnabled ? 0.001 ether : 0;

        return baseCost + cpuCost + memoryCost + storageCost + networkCost + gpuCost;
    }

    /**
     * @notice Calculate required capacity units
     * @param config VM configuration
     * @return capacity Required capacity units
     */
    function calculateRequiredCapacity(VMConfig memory config) public pure returns (uint256) {
        return config.cpuCores + (config.memoryGB / 4) + (config.storageGB / 100);
    }

    /**
     * @notice Get all VMs for a user
     * @param user User address
     * @return vmIds Array of VM identifiers
     */
    function getUserVMs(address user) external view returns (bytes32[] memory) {
        return userVMs[user].values();
    }

    /**
     * @notice Get all VMs for a provider
     * @param provider Provider address
     * @return vmIds Array of VM identifiers
     */
    function getProviderVMs(address provider) external view returns (bytes32[] memory) {
        return providerVMs[provider].values();
    }
}

// Resource Token Contract (ERC20)
contract ResourceToken is ERC20, AccessControl {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");

    // Resource metrics
    struct ResourceMetrics {
        uint256 cpuSeconds;
        uint256 memoryGBHours;
        uint256 storageGBDays;
        uint256 networkGBTransferred;
    }

    mapping(address => ResourceMetrics) public userMetrics;

    constructor() ERC20("NovaCron Resource Token", "NCR") {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(MINTER_ROLE, msg.sender);
    }

    /**
     * @notice Mint resource tokens based on usage
     * @param account User address
     * @param amount Token amount
     * @param metrics Resource metrics
     */
    function mintResourceTokens(
        address account,
        uint256 amount,
        ResourceMetrics memory metrics
    ) external onlyRole(MINTER_ROLE) {
        _mint(account, amount);

        userMetrics[account].cpuSeconds += metrics.cpuSeconds;
        userMetrics[account].memoryGBHours += metrics.memoryGBHours;
        userMetrics[account].storageGBDays += metrics.storageGBDays;
        userMetrics[account].networkGBTransferred += metrics.networkGBTransferred;
    }

    /**
     * @notice Burn resource tokens for service payment
     * @param account User address
     * @param amount Token amount
     */
    function burnResourceTokens(address account, uint256 amount)
        external
        onlyRole(BURNER_ROLE)
    {
        _burn(account, amount);
    }
}

// VM NFT Contract (ERC721)
contract VMNFT is ERC721, AccessControl {
    using EnumerableSet for EnumerableSet.UintSet;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    struct VMNFTMetadata {
        bytes32 vmId;
        string name;
        string imageURI;
        VMConfig config;
        uint256 createdAt;
        address originalProvider;
        uint256[] migrationHistory;
    }

    uint256 private _tokenIdCounter;
    mapping(uint256 => VMNFTMetadata) public tokenMetadata;
    mapping(bytes32 => uint256) public vmToToken;

    constructor() ERC721("NovaCron VM", "NCVM") {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(MINTER_ROLE, msg.sender);
    }

    /**
     * @notice Mint VM NFT
     * @param to Owner address
     * @param vmId VM identifier
     * @param config VM configuration
     * @return tokenId NFT token ID
     */
    function mintVMNFT(
        address to,
        bytes32 vmId,
        VMConfig memory config,
        string memory name,
        string memory imageURI
    ) external onlyRole(MINTER_ROLE) returns (uint256) {
        uint256 tokenId = _tokenIdCounter++;
        _safeMint(to, tokenId);

        tokenMetadata[tokenId] = VMNFTMetadata({
            vmId: vmId,
            name: name,
            imageURI: imageURI,
            config: config,
            createdAt: block.timestamp,
            originalProvider: msg.sender,
            migrationHistory: new uint256[](0)
        });

        vmToToken[vmId] = tokenId;

        return tokenId;
    }

    /**
     * @notice Record VM migration in NFT
     * @param tokenId NFT token ID
     * @param timestamp Migration timestamp
     */
    function recordMigration(uint256 tokenId, uint256 timestamp)
        external
        onlyRole(MINTER_ROLE)
    {
        tokenMetadata[tokenId].migrationHistory.push(timestamp);
    }

    /**
     * @notice Get migration history
     * @param tokenId NFT token ID
     * @return history Migration timestamps
     */
    function getMigrationHistory(uint256 tokenId)
        external
        view
        returns (uint256[] memory)
    {
        return tokenMetadata[tokenId].migrationHistory;
    }
}

// Consensus Oracle Contract
contract ConsensusOracle is AccessControl, ReentrancyGuard {
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");

    struct ConsensusData {
        bytes32 dataHash;
        uint256 timestamp;
        address[] validators;
        mapping(address => bool) signatures;
        uint256 signatureCount;
        bool finalized;
    }

    struct ValidationRequest {
        bytes32 requestId;
        bytes32 vmId;
        string requestType;
        bytes data;
        uint256 timestamp;
        uint256 requiredValidators;
        mapping(address => bytes) validatorResponses;
        uint256 responseCount;
        bool completed;
    }

    mapping(bytes32 => ConsensusData) public consensusData;
    mapping(bytes32 => ValidationRequest) public validationRequests;

    uint256 public requiredValidators = 3;
    uint256 public validatorStakeRequired = 5 ether;

    event ConsensusReached(bytes32 indexed dataHash, uint256 validatorCount);
    event ValidationRequested(bytes32 indexed requestId, bytes32 vmId, string requestType);
    event ValidationCompleted(bytes32 indexed requestId, bool success);

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Register as a validator
     */
    function registerValidator() external payable {
        require(msg.value >= validatorStakeRequired, "Insufficient stake");
        _setupRole(ORACLE_ROLE, msg.sender);
    }

    /**
     * @notice Submit data for consensus
     * @param dataHash Hash of the data
     * @param signature Validator signature
     */
    function submitConsensusData(bytes32 dataHash, bytes memory signature)
        external
        onlyRole(ORACLE_ROLE)
    {
        ConsensusData storage data = consensusData[dataHash];

        if (data.timestamp == 0) {
            data.dataHash = dataHash;
            data.timestamp = block.timestamp;
        }

        require(!data.signatures[msg.sender], "Already signed");
        require(!data.finalized, "Already finalized");

        // Verify signature
        bytes32 messageHash = keccak256(abi.encodePacked(dataHash, block.chainid));
        address signer = messageHash.toEthSignedMessageHash().recover(signature);
        require(signer == msg.sender, "Invalid signature");

        data.validators.push(msg.sender);
        data.signatures[msg.sender] = true;
        data.signatureCount++;

        if (data.signatureCount >= requiredValidators) {
            data.finalized = true;
            emit ConsensusReached(dataHash, data.signatureCount);
        }
    }

    /**
     * @notice Request validation for VM operation
     * @param vmId VM identifier
     * @param requestType Type of validation
     * @param requestData Request data
     * @return requestId Validation request ID
     */
    function requestValidation(
        bytes32 vmId,
        string memory requestType,
        bytes memory requestData
    ) external returns (bytes32) {
        bytes32 requestId = keccak256(
            abi.encodePacked(vmId, requestType, requestData, block.timestamp)
        );

        ValidationRequest storage request = validationRequests[requestId];
        request.requestId = requestId;
        request.vmId = vmId;
        request.requestType = requestType;
        request.data = requestData;
        request.timestamp = block.timestamp;
        request.requiredValidators = requiredValidators;

        emit ValidationRequested(requestId, vmId, requestType);

        return requestId;
    }

    /**
     * @notice Submit validation response
     * @param requestId Validation request ID
     * @param response Validator response
     */
    function submitValidationResponse(bytes32 requestId, bytes memory response)
        external
        onlyRole(ORACLE_ROLE)
    {
        ValidationRequest storage request = validationRequests[requestId];
        require(request.timestamp > 0, "Request does not exist");
        require(!request.completed, "Request already completed");
        require(
            request.validatorResponses[msg.sender].length == 0,
            "Already responded"
        );

        request.validatorResponses[msg.sender] = response;
        request.responseCount++;

        if (request.responseCount >= request.requiredValidators) {
            request.completed = true;

            // Process consensus on responses
            bool success = processValidationConsensus(requestId);
            emit ValidationCompleted(requestId, success);
        }
    }

    /**
     * @notice Process validation consensus
     * @param requestId Validation request ID
     * @return success Whether consensus was reached
     */
    function processValidationConsensus(bytes32 requestId)
        internal
        returns (bool)
    {
        // Implementation for processing validator responses
        // and determining consensus
        return true;
    }
}

// Governance Contract
contract NovaCronGovernance is AccessControl {
    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant VOTER_ROLE = keccak256("VOTER_ROLE");

    struct Proposal {
        uint256 id;
        address proposer;
        string title;
        string description;
        bytes callData;
        address targetContract;
        uint256 startTime;
        uint256 endTime;
        uint256 forVotes;
        uint256 againstVotes;
        bool executed;
        mapping(address => bool) hasVoted;
    }

    uint256 public proposalCount;
    mapping(uint256 => Proposal) public proposals;

    uint256 public constant VOTING_PERIOD = 3 days;
    uint256 public constant EXECUTION_DELAY = 2 days;
    uint256 public quorumPercentage = 40; // 40%

    event ProposalCreated(uint256 indexed proposalId, address proposer, string title);
    event VoteCast(uint256 indexed proposalId, address voter, bool support, uint256 votes);
    event ProposalExecuted(uint256 indexed proposalId);

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Create a governance proposal
     * @param title Proposal title
     * @param description Proposal description
     * @param targetContract Target contract address
     * @param callData Encoded function call
     * @return proposalId Proposal ID
     */
    function createProposal(
        string memory title,
        string memory description,
        address targetContract,
        bytes memory callData
    ) external onlyRole(PROPOSER_ROLE) returns (uint256) {
        uint256 proposalId = proposalCount++;

        Proposal storage proposal = proposals[proposalId];
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.title = title;
        proposal.description = description;
        proposal.targetContract = targetContract;
        proposal.callData = callData;
        proposal.startTime = block.timestamp;
        proposal.endTime = block.timestamp + VOTING_PERIOD;

        emit ProposalCreated(proposalId, msg.sender, title);

        return proposalId;
    }

    /**
     * @notice Cast vote on proposal
     * @param proposalId Proposal ID
     * @param support Vote for or against
     */
    function castVote(uint256 proposalId, bool support)
        external
        onlyRole(VOTER_ROLE)
    {
        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.startTime, "Voting not started");
        require(block.timestamp <= proposal.endTime, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");

        uint256 votes = getVotingPower(msg.sender);

        if (support) {
            proposal.forVotes += votes;
        } else {
            proposal.againstVotes += votes;
        }

        proposal.hasVoted[msg.sender] = true;

        emit VoteCast(proposalId, msg.sender, support, votes);
    }

    /**
     * @notice Execute approved proposal
     * @param proposalId Proposal ID
     */
    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Already executed");
        require(
            block.timestamp >= proposal.endTime + EXECUTION_DELAY,
            "Execution delay not met"
        );

        uint256 totalVotes = proposal.forVotes + proposal.againstVotes;
        uint256 quorum = (getTotalVotingPower() * quorumPercentage) / 100;

        require(totalVotes >= quorum, "Quorum not reached");
        require(proposal.forVotes > proposal.againstVotes, "Proposal not passed");

        proposal.executed = true;

        // Execute proposal
        (bool success, ) = proposal.targetContract.call(proposal.callData);
        require(success, "Execution failed");

        emit ProposalExecuted(proposalId);
    }

    /**
     * @notice Get voting power of an address
     * @param voter Voter address
     * @return power Voting power
     */
    function getVotingPower(address voter) public view returns (uint256) {
        // Implementation based on token holdings or other metrics
        return 1;
    }

    /**
     * @notice Get total voting power
     * @return total Total voting power
     */
    function getTotalVotingPower() public view returns (uint256) {
        // Implementation
        return 100;
    }
}