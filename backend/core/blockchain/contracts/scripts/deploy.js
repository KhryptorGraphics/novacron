const hre = require("hardhat");

async function main() {
  console.log("Deploying NovaCron blockchain contracts...");

  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());

  // Deploy VMLifecycle
  console.log("\nDeploying VMLifecycle...");
  const VMLifecycle = await ethers.getContractFactory("VMLifecycle");
  const vmLifecycle = await VMLifecycle.deploy();
  await vmLifecycle.deployed();
  console.log("VMLifecycle deployed to:", vmLifecycle.address);

  // Deploy ResourceMarket
  console.log("\nDeploying ResourceMarket...");
  const ResourceMarket = await ethers.getContractFactory("ResourceMarket");
  const resourceMarket = await ResourceMarket.deploy();
  await resourceMarket.deployed();
  console.log("ResourceMarket deployed to:", resourceMarket.address);

  // Deploy SLAContract
  console.log("\nDeploying SLAContract...");
  const SLAContract = await ethers.getContractFactory("SLAContract");
  const slaContract = await SLAContract.deploy();
  await slaContract.deployed();
  console.log("SLAContract deployed to:", slaContract.address);

  // Save deployment addresses
  const deploymentInfo = {
    network: hre.network.name,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: {
      VMLifecycle: vmLifecycle.address,
      ResourceMarket: resourceMarket.address,
      SLAContract: slaContract.address
    }
  };

  console.log("\n=== Deployment Summary ===");
  console.log(JSON.stringify(deploymentInfo, null, 2));

  // Save to file
  const fs = require("fs");
  const path = require("path");
  const deploymentsDir = path.join(__dirname, "../deployments");

  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  const filename = path.join(deploymentsDir, `${hre.network.name}.json`);
  fs.writeFileSync(filename, JSON.stringify(deploymentInfo, null, 2));
  console.log(`\nDeployment info saved to: ${filename}`);

  // Verification instructions
  console.log("\n=== Verification Commands ===");
  console.log(`npx hardhat verify --network ${hre.network.name} ${vmLifecycle.address}`);
  console.log(`npx hardhat verify --network ${hre.network.name} ${resourceMarket.address}`);
  console.log(`npx hardhat verify --network ${hre.network.name} ${slaContract.address}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
