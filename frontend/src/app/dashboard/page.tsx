"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import UnifiedDashboard from "@/components/dashboard/UnifiedDashboard";

export default function DashboardPage() {
  return <UnifiedDashboard />;
}