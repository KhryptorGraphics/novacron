// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title ResourceMarket
 * @dev Automated market maker for tokenized resources
 */
contract ResourceMarket is ReentrancyGuard {

    enum ResourceType { CPU, Memory, Storage, Network }

    struct Pool {
        IERC20 token;
        uint256 reserve;
        uint256 totalLiquidity;
        mapping(address => uint256) liquidityProviders;
    }

    mapping(ResourceType => Pool) public pools;

    // Bonding curve parameters
    uint256 public constant CURVE_STEEPNESS = 1000;
    uint256 public constant BASE_PRICE = 1e15; // 0.001 ETH

    event LiquidityAdded(ResourceType resourceType, address indexed provider, uint256 amount);
    event LiquidityRemoved(ResourceType resourceType, address indexed provider, uint256 amount);
    event ResourcePurchased(ResourceType resourceType, address indexed buyer, uint256 amount, uint256 cost);
    event ResourceSold(ResourceType resourceType, address indexed seller, uint256 amount, uint256 revenue);

    /**
     * @dev Add liquidity to resource pool
     */
    function addLiquidity(ResourceType resourceType, uint256 amount)
        public
        payable
        nonReentrant
    {
        Pool storage pool = pools[resourceType];

        require(amount > 0, "Amount must be positive");
        require(msg.value > 0, "Must send ETH");

        // Transfer tokens from provider
        require(
            pool.token.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );

        pool.reserve += amount;
        pool.liquidityProviders[msg.sender] += amount;
        pool.totalLiquidity += amount;

        emit LiquidityAdded(resourceType, msg.sender, amount);
    }

    /**
     * @dev Remove liquidity from resource pool
     */
    function removeLiquidity(ResourceType resourceType, uint256 amount)
        public
        nonReentrant
    {
        Pool storage pool = pools[resourceType];

        require(amount > 0, "Amount must be positive");
        require(
            pool.liquidityProviders[msg.sender] >= amount,
            "Insufficient liquidity"
        );

        pool.liquidityProviders[msg.sender] -= amount;
        pool.reserve -= amount;
        pool.totalLiquidity -= amount;

        // Transfer tokens back to provider
        require(
            pool.token.transfer(msg.sender, amount),
            "Token transfer failed"
        );

        emit LiquidityRemoved(resourceType, msg.sender, amount);
    }

    /**
     * @dev Purchase resources using bonding curve pricing
     */
    function purchaseResource(ResourceType resourceType, uint256 amount)
        public
        payable
        nonReentrant
    {
        uint256 cost = calculatePurchaseCost(resourceType, amount);
        require(msg.value >= cost, "Insufficient payment");

        Pool storage pool = pools[resourceType];
        require(pool.reserve >= amount, "Insufficient reserve");

        pool.reserve -= amount;

        // Transfer tokens to buyer
        require(
            pool.token.transfer(msg.sender, amount),
            "Token transfer failed"
        );

        // Refund excess payment
        if (msg.value > cost) {
            payable(msg.sender).transfer(msg.value - cost);
        }

        emit ResourcePurchased(resourceType, msg.sender, amount, cost);
    }

    /**
     * @dev Sell resources back to pool
     */
    function sellResource(ResourceType resourceType, uint256 amount)
        public
        nonReentrant
    {
        uint256 revenue = calculateSaleRevenue(resourceType, amount);

        Pool storage pool = pools[resourceType];

        // Transfer tokens from seller
        require(
            pool.token.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );

        pool.reserve += amount;

        // Pay seller
        payable(msg.sender).transfer(revenue);

        emit ResourceSold(resourceType, msg.sender, amount, revenue);
    }

    /**
     * @dev Calculate purchase cost using bonding curve
     * Price = BASE_PRICE * (1 + reserve / CURVE_STEEPNESS)
     */
    function calculatePurchaseCost(ResourceType resourceType, uint256 amount)
        public
        view
        returns (uint256)
    {
        Pool storage pool = pools[resourceType];
        uint256 currentReserve = pool.reserve;

        // Integral of bonding curve from currentReserve to (currentReserve - amount)
        uint256 cost = BASE_PRICE * amount;
        uint256 curveCost = (BASE_PRICE * currentReserve * amount) / CURVE_STEEPNESS;

        return cost + curveCost;
    }

    /**
     * @dev Calculate sale revenue using bonding curve
     */
    function calculateSaleRevenue(ResourceType resourceType, uint256 amount)
        public
        view
        returns (uint256)
    {
        Pool storage pool = pools[resourceType];
        uint256 currentReserve = pool.reserve;

        // Integral of bonding curve from currentReserve to (currentReserve + amount)
        uint256 revenue = BASE_PRICE * amount;
        uint256 curveRevenue = (BASE_PRICE * (currentReserve + amount) * amount) / CURVE_STEEPNESS;

        return revenue - curveRevenue;
    }

    /**
     * @dev Get current spot price for resource
     */
    function getSpotPrice(ResourceType resourceType)
        public
        view
        returns (uint256)
    {
        Pool storage pool = pools[resourceType];
        return BASE_PRICE * (CURVE_STEEPNESS + pool.reserve) / CURVE_STEEPNESS;
    }

    /**
     * @dev Get pool information
     */
    function getPoolInfo(ResourceType resourceType)
        public
        view
        returns (uint256 reserve, uint256 totalLiquidity, uint256 spotPrice)
    {
        Pool storage pool = pools[resourceType];
        return (
            pool.reserve,
            pool.totalLiquidity,
            getSpotPrice(resourceType)
        );
    }

    /**
     * @dev Get liquidity provider balance
     */
    function getLiquidityProviderBalance(ResourceType resourceType, address provider)
        public
        view
        returns (uint256)
    {
        return pools[resourceType].liquidityProviders[provider];
    }
}
