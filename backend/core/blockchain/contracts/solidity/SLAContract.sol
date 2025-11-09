// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SLAContract
 * @dev Smart contract for SLA enforcement with automatic penalties
 */
contract SLAContract {

    enum SLAStatus { Active, Violated, Completed, Cancelled }

    struct SLA {
        uint256 slaId;
        address provider;
        address customer;
        uint256 startDate;
        uint256 endDate;
        uint256 stake;
        SLAStatus status;
        mapping(string => uint256) guarantees; // "uptime" => 999 (99.9%)
        mapping(string => uint256) penalties; // "uptime" => 1000 (penalty amount)
        mapping(string => uint256) violations; // "uptime" => 5 (violation count)
    }

    mapping(uint256 => SLA) public slas;
    uint256 public slaCount;

    event SLACreated(uint256 indexed slaId, address indexed provider, address indexed customer);
    event SLAViolation(uint256 indexed slaId, string guarantee, uint256 penalty);
    event PenaltyEnforced(uint256 indexed slaId, address indexed customer, uint256 amount);
    event SLACompleted(uint256 indexed slaId);
    event StakeReturned(uint256 indexed slaId, address indexed provider, uint256 amount);

    modifier onlyProvider(uint256 slaId) {
        require(slas[slaId].provider == msg.sender, "Not SLA provider");
        _;
    }

    modifier onlyCustomer(uint256 slaId) {
        require(slas[slaId].customer == msg.sender, "Not SLA customer");
        _;
    }

    modifier slaActive(uint256 slaId) {
        require(slas[slaId].status == SLAStatus.Active, "SLA not active");
        require(block.timestamp <= slas[slaId].endDate, "SLA expired");
        _;
    }

    /**
     * @dev Create a new SLA contract
     */
    function createSLA(
        address customer,
        uint256 duration,
        string[] memory guaranteeNames,
        uint256[] memory guaranteeValues,
        uint256[] memory penaltyAmounts
    ) public payable returns (uint256) {
        require(msg.value > 0, "Must stake collateral");
        require(
            guaranteeNames.length == guaranteeValues.length &&
            guaranteeValues.length == penaltyAmounts.length,
            "Array length mismatch"
        );

        slaCount++;
        uint256 slaId = slaCount;

        SLA storage sla = slas[slaId];
        sla.slaId = slaId;
        sla.provider = msg.sender;
        sla.customer = customer;
        sla.startDate = block.timestamp;
        sla.endDate = block.timestamp + duration;
        sla.stake = msg.value;
        sla.status = SLAStatus.Active;

        // Set guarantees and penalties
        for (uint256 i = 0; i < guaranteeNames.length; i++) {
            sla.guarantees[guaranteeNames[i]] = guaranteeValues[i];
            sla.penalties[guaranteeNames[i]] = penaltyAmounts[i];
        }

        emit SLACreated(slaId, msg.sender, customer);

        return slaId;
    }

    /**
     * @dev Report SLA violation and enforce penalty
     */
    function reportViolation(uint256 slaId, string memory guarantee, uint256 actualValue)
        public
        onlyCustomer(slaId)
        slaActive(slaId)
    {
        SLA storage sla = slas[slaId];
        uint256 guaranteedValue = sla.guarantees[guarantee];

        require(guaranteedValue > 0, "Guarantee not found");
        require(actualValue < guaranteedValue, "No violation");

        sla.violations[guarantee]++;

        emit SLAViolation(slaId, guarantee, sla.penalties[guarantee]);

        // Automatically enforce penalty
        enforcePenalty(slaId, guarantee);
    }

    /**
     * @dev Automatically enforce penalty for violation
     */
    function enforcePenalty(uint256 slaId, string memory guarantee)
        internal
    {
        SLA storage sla = slas[slaId];
        uint256 penaltyAmount = sla.penalties[guarantee];

        require(sla.stake >= penaltyAmount, "Insufficient stake");

        sla.stake -= penaltyAmount;

        // Transfer penalty to customer
        payable(sla.customer).transfer(penaltyAmount);

        emit PenaltyEnforced(slaId, sla.customer, penaltyAmount);

        // Mark as violated if stake depleted
        if (sla.stake == 0) {
            sla.status = SLAStatus.Violated;
        }
    }

    /**
     * @dev Complete SLA and return remaining stake
     */
    function completeSLA(uint256 slaId)
        public
        onlyProvider(slaId)
    {
        SLA storage sla = slas[slaId];
        require(block.timestamp > sla.endDate, "SLA not yet ended");
        require(sla.status == SLAStatus.Active, "SLA not active");

        sla.status = SLAStatus.Completed;

        // Return remaining stake to provider
        if (sla.stake > 0) {
            uint256 stakeAmount = sla.stake;
            sla.stake = 0;
            payable(sla.provider).transfer(stakeAmount);

            emit StakeReturned(slaId, sla.provider, stakeAmount);
        }

        emit SLACompleted(slaId);
    }

    /**
     * @dev Get SLA details
     */
    function getSLA(uint256 slaId)
        public
        view
        returns (
            address provider,
            address customer,
            uint256 startDate,
            uint256 endDate,
            uint256 stake,
            SLAStatus status
        )
    {
        SLA storage sla = slas[slaId];
        return (
            sla.provider,
            sla.customer,
            sla.startDate,
            sla.endDate,
            sla.stake,
            sla.status
        );
    }

    /**
     * @dev Get guarantee value
     */
    function getGuarantee(uint256 slaId, string memory guarantee)
        public
        view
        returns (uint256)
    {
        return slas[slaId].guarantees[guarantee];
    }

    /**
     * @dev Get penalty amount
     */
    function getPenalty(uint256 slaId, string memory guarantee)
        public
        view
        returns (uint256)
    {
        return slas[slaId].penalties[guarantee];
    }

    /**
     * @dev Get violation count
     */
    function getViolationCount(uint256 slaId, string memory guarantee)
        public
        view
        returns (uint256)
    {
        return slas[slaId].violations[guarantee];
    }
}
