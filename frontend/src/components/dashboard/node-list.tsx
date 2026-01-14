"use client";

import React from "react";

interface Node {
  id: string;
  name: string;
  role: string;
  state: string;
  address: string;
  port: number;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  vm_count: number;
  joined_at: string;
}

interface NodeListProps {
  nodes: Node[];
  onAction: (nodeId: string, action: string) => void;
}

export function NodeList({ nodes, onAction }: NodeListProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="text-xs uppercase bg-gray-100 dark:bg-gray-700">
          <tr>
            <th scope="col" className="px-6 py-3">Name</th>
            <th scope="col" className="px-6 py-3">Role</th>
            <th scope="col" className="px-6 py-3">Status</th>
            <th scope="col" className="px-6 py-3">Address</th>
            <th scope="col" className="px-6 py-3">CPU</th>
            <th scope="col" className="px-6 py-3">Memory</th>
            <th scope="col" className="px-6 py-3">VMs</th>
            <th scope="col" className="px-6 py-3">Actions</th>
          </tr>
        </thead>
        <tbody>
          {nodes.map((node) => (
            <tr key={node.id} className="border-b dark:border-gray-700">
              <td className="px-6 py-4 font-medium">{node.name}</td>
              <td className="px-6 py-4">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  node.role === 'master' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300' : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
                }`}>
                  {node.role}
                </span>
              </td>
              <td className="px-6 py-4">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  node.state === 'running' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
                }`}>
                  {node.state}
                </span>
              </td>
              <td className="px-6 py-4">{node.address}:{node.port}</td>
              <td className="px-6 py-4">
                <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(100, node.cpu_usage)}%` }}
                  ></div>
                </div>
                <span className="text-xs mt-1">{node.cpu_usage.toFixed(1)}%</span>
              </td>
              <td className="px-6 py-4">
                <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                  <div 
                    className="bg-green-600 h-2.5 rounded-full" 
                    style={{ width: `${Math.min(100, (node.memory_usage / 1024 / 1024 / 1024) * 10)}%` }}
                  ></div>
                </div>
                <span className="text-xs mt-1">{(node.memory_usage / 1024 / 1024 / 1024).toFixed(1)} GB</span>
              </td>
              <td className="px-6 py-4">{node.vm_count}</td>
              <td className="px-6 py-4">
                <div className="flex space-x-2">
                  <button 
                    onClick={() => onAction(node.id, 'restart')}
                    className="text-blue-600 hover:text-blue-900 dark:text-blue-400 dark:hover:text-blue-200"
                  >
                    Restart
                  </button>
                  {node.state === 'running' ? (
                    <button 
                      onClick={() => onAction(node.id, 'stop')}
                      className="text-red-600 hover:text-red-900 dark:text-red-400 dark:hover:text-red-200"
                    >
                      Stop
                    </button>
                  ) : (
                    <button 
                      onClick={() => onAction(node.id, 'start')}
                      className="text-green-600 hover:text-green-900 dark:text-green-400 dark:hover:text-green-200"
                    >
                      Start
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
          {nodes.length === 0 && (
            <tr>
              <td colSpan={8} className="px-6 py-4 text-center">No nodes found</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
