"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import Link from "next/link";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";

import { useVM } from "@/lib/api/hooks/useVM";
import { useVMAction } from "@/lib/api/hooks/useVMAction";
import type { VM } from "@/lib/api/types";
import { useToast } from "@/components/ui/use-toast";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function VMDetailPage() {
  const params = useParams<{ id: string }>();
  const id = typeof params?.id === "string" ? params.id : Array.isArray(params?.id) ? params?.id?.[0] : "";
  const { vm, isLoading, error } = useVM(id);
  const { toast } = useToast();
  const qc = useQueryClient();

  const [role, setRole] = useState<"viewer"|"operator">("viewer");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("coreRole");
      if (saved === "viewer" || saved === "operator") setRole(saved as any);
    } catch {}
  }, []);

  const disabled = role !== "operator";

  const start = useVMAction(id, "start", { role });
  const stop = useVMAction(id, "stop", { role });
  const restart = useVMAction(id, "restart", { role });
  const pause = useVMAction(id, "pause", { role });
  const resume = useVMAction(id, "resume", { role });

  useEffect(() => {
    if (start.isSuccess) { toast({ title: "Start requested" }); qc.invalidateQueries({ queryKey: ["vm", id] }); qc.invalidateQueries({ queryKey: ["vms"] }); }
    if (stop.isSuccess) { toast({ title: "Stop requested" }); qc.invalidateQueries({ queryKey: ["vm", id] }); qc.invalidateQueries({ queryKey: ["vms"] }); }
    if (restart.isSuccess) { toast({ title: "Restart requested" }); qc.invalidateQueries({ queryKey: ["vm", id] }); qc.invalidateQueries({ queryKey: ["vms"] }); }
    if (pause.isSuccess) { toast({ title: "Pause requested" }); qc.invalidateQueries({ queryKey: ["vm", id] }); qc.invalidateQueries({ queryKey: ["vms"] }); }
    if (resume.isSuccess) { toast({ title: "Resume requested" }); qc.invalidateQueries({ queryKey: ["vm", id] }); qc.invalidateQueries({ queryKey: ["vms"] }); }
  }, [start.isSuccess, stop.isSuccess, restart.isSuccess, pause.isSuccess, resume.isSuccess, id, toast, qc]);

  useEffect(() => {
    if (start.isError) toast({ title: "Start failed", description: String((start.error as any)?.message ?? start.error), variant: "destructive" as any });
    if (stop.isError) toast({ title: "Stop failed", description: String((stop.error as any)?.message ?? stop.error), variant: "destructive" as any });
    if (restart.isError) toast({ title: "Restart failed", description: String((restart.error as any)?.message ?? restart.error), variant: "destructive" as any });
    if (pause.isError) toast({ title: "Pause failed", description: String((pause.error as any)?.message ?? pause.error), variant: "destructive" as any });
    if (resume.isError) toast({ title: "Resume failed", description: String((resume.error as any)?.message ?? resume.error), variant: "destructive" as any });
  }, [start.isError, stop.isError, restart.isError, pause.isError, resume.isError, start.error, stop.error, restart.error, pause.error, resume.error, toast]);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>VM Detail</CardTitle>
          <CardDescription>
            <Link href="/core/vms" className="underline">Back to list</Link>
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div>Loading VM...</div>
          ) : error ? (
            <div className="text-red-500">Failed to load VM: {String((error as any)?.message ?? error)}</div>
          ) : !vm ? (
            <div>VM not found.</div>
          ) : (
            <div className="space-y-4">
              <div className="text-sm text-muted-foreground">ID</div>
              <div className="font-mono text-xs break-all">{vm.id}</div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">Name</div>
                  <div>{vm.name}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">State</div>
                  <div><Badge>{vm.state}</Badge></div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Node</div>
                  <div>{vm.node_id ?? "-"}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Created</div>
                  <div>{vm.created_at}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Updated</div>
                  <div>{vm.updated_at}</div>
                </div>
              </div>
              {"tags" in (vm as any) && (vm as any).tags ? (
                <div>
                  <div className="text-sm text-muted-foreground">Tags</div>
                  <ul className="list-disc ml-6">
                    {Object.entries((vm as any).tags as Record<string,string>).map(([k,v]) => (
                      <li key={k}><span className="font-medium">{k}:</span> {v}</li>
                    ))}
                  </ul>
                </div>
              ) : null}

              <div className="flex gap-2">
                <Button size="sm" variant="secondary" onClick={()=>start.mutate()} disabled={disabled || start.isPending}>Start</Button>
                <Button size="sm" variant="secondary" onClick={()=>stop.mutate()} disabled={disabled || stop.isPending}>Stop</Button>
                <Button size="sm" variant="secondary" onClick={()=>restart.mutate()} disabled={disabled || restart.isPending}>Restart</Button>
                <Button size="sm" variant="secondary" onClick={()=>pause.mutate()} disabled={disabled || pause.isPending}>Pause</Button>
                <Button size="sm" variant="secondary" onClick={()=>resume.mutate()} disabled={disabled || resume.isPending}>Resume</Button>
                {disabled && <div className="text-xs text-muted-foreground self-center">Switch to operator on the list page to enable actions</div>}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

