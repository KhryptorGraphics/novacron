"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import MonitoringDashboard from "@/components/monitoring/MonitoringDashboard";

export default function MonitoringPage() {
  return <MonitoringDashboard />;
}