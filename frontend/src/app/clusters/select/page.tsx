"use client";

export const dynamic = 'force-dynamic';

import { useState } from "react";
import { useRouter } from "next/navigation";
import AuthGuard from "@/components/auth/AuthGuard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { useToast } from "@/components/ui/use-toast";
import { useAuth } from "@/hooks/useAuth";

export default function ClusterSelectionPage() {
  const router = useRouter();
  const { toast } = useToast();
  const { memberships, selectedCluster, selectCluster } = useAuth();
  const [submittingClusterId, setSubmittingClusterId] = useState<string | null>(null);

  const activeMemberships = memberships.filter((membership) => membership.admitted || membership.state === 'active');

  async function handleSelect(clusterId: string) {
    setSubmittingClusterId(clusterId);
    try {
      const destination = await selectCluster(clusterId);
      toast({
        title: "Cluster Selected",
        description: "Your runtime session is now bound to the selected cluster.",
      });
      router.push(destination);
    } catch (error) {
      console.error('Failed to select cluster:', error);
      toast({
        title: "Selection Failed",
        description: "NovaCron could not bind your session to that cluster.",
        variant: "destructive",
      });
    } finally {
      setSubmittingClusterId(null);
    }
  }

  return (
    <AuthGuard>
      <main className="container mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-4 py-12">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">Choose a Cluster</h1>
          <p className="mt-2 text-muted-foreground">
            NovaCron ranked your eligible clusters by current tier, interconnect quality, and any measured edge performance.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {activeMemberships.map((membership) => {
            const cluster = membership.cluster;
            if (!cluster) {
              return null;
            }

            const isSelected = selectedCluster?.id === cluster.id || membership.selected;
            const isSubmitting = submittingClusterId === cluster.id;

            return (
              <Card key={cluster.id} className={isSelected ? 'border-primary shadow-sm' : ''}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between gap-4">
                    <span>{cluster.name}</span>
                    <span className="rounded-full bg-muted px-3 py-1 text-xs font-medium uppercase tracking-wide">
                      {cluster.tier}
                    </span>
                  </CardTitle>
                  <CardDescription>
                    Cluster ID: {cluster.id}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  <div className="flex items-center justify-between">
                    <span>Performance Score</span>
                    <span className="font-medium text-foreground">{cluster.performanceScore.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Interconnect Latency</span>
                    <span className="font-medium text-foreground">{cluster.interconnectLatencyMs} ms</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Interconnect Bandwidth</span>
                    <span className="font-medium text-foreground">{cluster.interconnectBandwidthMbps} Mbps</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Nodes</span>
                    <span className="font-medium text-foreground">
                      {cluster.currentNodeCount} / {cluster.maxSupportedNodeCount}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Growth State</span>
                    <span className="font-medium capitalize text-foreground">{cluster.growthState.replace(/_/g, ' ')}</span>
                  </div>
                  {typeof cluster.edgeLatencyMs === 'number' && typeof cluster.edgeBandwidthMbps === 'number' && (
                    <div className="rounded-md border bg-muted/30 p-3">
                      <div className="font-medium text-foreground">Your Edge Metrics</div>
                      <div className="mt-1 text-xs">
                        {cluster.edgeLatencyMs} ms latency, {cluster.edgeBandwidthMbps} Mbps bandwidth
                      </div>
                    </div>
                  )}
                </CardContent>
                <CardFooter>
                  <Button
                    className="w-full"
                    disabled={isSelected || isSubmitting}
                    onClick={() => handleSelect(cluster.id)}
                  >
                    {isSubmitting && <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />}
                    {isSelected ? 'Selected' : 'Use This Cluster'}
                  </Button>
                </CardFooter>
              </Card>
            );
          })}
        </div>
      </main>
    </AuthGuard>
  );
}
