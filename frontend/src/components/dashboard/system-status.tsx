"use client";

import React from "react";

interface SystemStatusProps {
  total_nodes: number;
  total_vms: number;
  total_cpu_usage: number;
  total_memory_usage: number;
  total_disk_usage: number;
}

export function SystemStatus({ 
  total_nodes, 
  total_vms, 
  total_cpu_usage, 
  total_memory_usage, 
  total_disk_usage 
}: SystemStatusProps) {
  
  // Convert bytes to GB for better readability
  const memoryGB = (total_memory_usage / 1024 / 1024 / 1024).toFixed(2);
  const diskGB = (total_disk_usage / 1024 / 1024 / 1024).toFixed(2);
  
  // Calculate the percentage of system resources used
  const cpuPercent = total_cpu_usage.toFixed(1);
  const memoryPercentage = ((total_memory_usage / (16 * 1024 * 1024 * 1024)) * 100).toFixed(1); // Assuming 16GB total system memory
  const diskPercentage = ((total_disk_usage / (1000 * 1024 * 1024 * 1024)) * 100).toFixed(1); // Assuming 1TB total system storage
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">System Status</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Nodes</h3>
          <p className="text-2xl font-bold">{total_nodes}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Hypervisor Instances</p>
        </div>
        
        <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">VMs</h3>
          <p className="text-2xl font-bold">{total_vms}</p>
          <p className="text-sm text-gray-500 dark:text-gray-400">Virtual Machines</p>
        </div>
        
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">CPU</h3>
          <div className="flex items-center mt-1">
            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mr-2">
              <div 
                className="bg-green-600 h-2.5 rounded-full" 
                style={{ width: `${Math.min(100, total_cpu_usage)}%` }}
              ></div>
            </div>
            <span>{cpuPercent}%</span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">Total Usage</p>
        </div>
        
        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Memory</h3>
          <div className="flex items-center mt-1">
            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mr-2">
              <div 
                className="bg-yellow-600 h-2.5 rounded-full" 
                style={{ width: `${memoryPercentage}%` }}
              ></div>
            </div>
            <span>{memoryPercentage}%</span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">{memoryGB} GB Used</p>
        </div>
        
        <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">Disk</h3>
          <div className="flex items-center mt-1">
            <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 mr-2">
              <div 
                className="bg-red-600 h-2.5 rounded-full" 
                style={{ width: `${diskPercentage}%` }}
              ></div>
            </div>
            <span>{diskPercentage}%</span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">{diskGB} GB Used</p>
        </div>
      </div>
      
      <div className="mt-6 py-4 px-6 bg-gray-50 dark:bg-gray-700/30 rounded-lg">
        <div className="flex justify-between mb-2">
          <span className="text-sm text-gray-500 dark:text-gray-400">Last Update:</span>
          <span className="text-sm">{new Date().toLocaleTimeString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-sm text-gray-500 dark:text-gray-400">System Health:</span>
          <span className="text-sm text-green-600 dark:text-green-400">
            <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-1"></span>
            Optimal
          </span>
        </div>
      </div>
    </div>
  );
}
