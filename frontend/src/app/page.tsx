"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function HomePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  const handleGetStarted = () => {
    router.push("/dashboard");
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-blue-900 to-black text-white">
      <div className="container flex flex-col items-center justify-center gap-12 px-4 py-16 md:py-24 lg:py-32">
        {loading ? (
          <div className="flex flex-col items-center gap-4">
            <div className="h-16 w-16 animate-spin rounded-full border-4 border-t-blue-500"></div>
            <p className="text-lg font-medium">Initializing NovaCron...</p>
          </div>
        ) : (
          <>
            <div className="flex flex-col items-center text-center">
              <div className="mb-8 relative h-24 w-24 md:h-32 md:w-32">
                {/* Placeholder for logo */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="h-20 w-20 md:h-28 md:w-28 rounded-full bg-blue-500 animate-pulse"></div>
                </div>
              </div>
              <h1 className="text-5xl font-extrabold tracking-tight sm:text-[5rem]">
                Nova<span className="text-blue-500">Cron</span>
              </h1>
              <p className="text-xl md:text-2xl mt-4 max-w-2xl">
                Next-Generation Distributed Cloud Hypervisor
              </p>
              <p className="text-lg text-gray-300 mt-4 max-w-2xl">
                Process-based virtualization with automatic discovery and web-based management
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3 md:gap-8 max-w-5xl">
              <div className="flex flex-col items-center p-6 bg-blue-950/30 rounded-lg backdrop-blur-sm border border-blue-900/50">
                <div className="mb-3 h-12 w-12 rounded-full bg-blue-600 flex items-center justify-center">
                  <span className="text-2xl">🚀</span>
                </div>
                <h3 className="text-xl font-bold">Lightweight Virtualization</h3>
                <p className="text-center mt-2 text-gray-300">
                  Process isolation using Linux namespaces and resource control with cgroups
                </p>
              </div>
              <div className="flex flex-col items-center p-6 bg-blue-950/30 rounded-lg backdrop-blur-sm border border-blue-900/50">
                <div className="mb-3 h-12 w-12 rounded-full bg-blue-600 flex items-center justify-center">
                  <span className="text-2xl">🔍</span>
                </div>
                <h3 className="text-xl font-bold">Auto-Discovery</h3>
                <p className="text-center mt-2 text-gray-300">
                  Internet-aware automatic discovery and coordination between hypervisor instances
                </p>
              </div>
              <div className="flex flex-col items-center p-6 bg-blue-950/30 rounded-lg backdrop-blur-sm border border-blue-900/50">
                <div className="mb-3 h-12 w-12 rounded-full bg-blue-600 flex items-center justify-center">
                  <span className="text-2xl">📊</span>
                </div>
                <h3 className="text-xl font-bold">Advanced Monitoring</h3>
                <p className="text-center mt-2 text-gray-300">
                  Real-time resource usage tracking with anomaly detection
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <button
                onClick={handleGetStarted}
                className="px-8 py-3 text-lg font-medium bg-blue-600 hover:bg-blue-700 transition-colors rounded-md shadow-lg hover:shadow-xl"
              >
                Get Started
              </button>
              <button
                onClick={() => router.push("/admin")}
                className="px-8 py-3 text-lg font-medium bg-red-600 hover:bg-red-700 transition-colors rounded-md shadow-lg hover:shadow-xl"
              >
                Admin Panel
              </button>
              <a
                href="https://github.com/novacron/novacron"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-3 text-lg font-medium border border-blue-600 hover:bg-blue-900/30 transition-colors rounded-md"
              >
                GitHub
              </a>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
