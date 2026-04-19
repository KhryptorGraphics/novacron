"use client";

export const dynamic = 'force-dynamic';

import Link from "next/link";
import AuthGuard from "@/components/auth/AuthGuard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/hooks/useAuth";

export default function ClusterAccessPage() {
  const { memberships } = useAuth();

  return (
    <AuthGuard>
      <main className="container mx-auto flex min-h-screen max-w-3xl items-center justify-center px-4 py-12">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>Cluster Access Pending</CardTitle>
            <CardDescription>
              Your account is authenticated, but NovaCron has not granted active cluster access yet.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-muted-foreground">
            {memberships.length === 0 ? (
              <p>No memberships have been provisioned for this account.</p>
            ) : (
              <div className="space-y-2">
                <p>Current membership states:</p>
                <ul className="space-y-2">
                  {memberships.map((membership) => (
                    <li key={`${membership.clusterId}-${membership.state}`} className="rounded-md border p-3">
                      <div className="font-medium text-foreground">
                        {membership.cluster?.name || membership.clusterId || 'Unknown cluster'}
                      </div>
                      <div className="mt-1 capitalize">
                        {membership.state || 'pending'} access
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
          <CardFooter className="flex justify-between gap-3">
            <Button asChild variant="outline">
              <Link href="/auth/login">Return to Login</Link>
            </Button>
            <Button asChild>
              <Link href="/dashboard">Retry Dashboard</Link>
            </Button>
          </CardFooter>
        </Card>
      </main>
    </AuthGuard>
  );
}
