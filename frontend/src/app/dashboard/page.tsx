"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import UnifiedDashboard from "@/components/dashboard/UnifiedDashboard";
import AuthGuard from "@/components/auth/AuthGuard";

export default function DashboardPage() {
  return (
    <AuthGuard>
      <UnifiedDashboard />
    </AuthGuard>
  );
}