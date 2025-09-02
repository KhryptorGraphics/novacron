"use client";

import { useEffect, useState } from "react";

// Disable static generation for this page
export const dynamic = 'force-dynamic';
import { useVMs } from "@/lib/api/hooks/useVMs";
import { fetchAction } from "./fetchAction";

import { useVMAction } from "@/lib/api/hooks/useVMAction";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { connectEvents } from "@/lib/ws/client";
import type { VM } from "@/lib/api/types";

import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";

const roleOptions = ["viewer", "operator"] as const;

type Role = (typeof roleOptions)[number];

export default function CoreVMsPage() {
  const [q, setQ] = useState("");
  const [state, setState] = useState<string>("");
  const [welcome, setWelcome] = useState<unknown>(null);
  const [role, setRole] = useState<Role>("viewer");
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(10);

  // reset to first page when filters change
  useEffect(() => { setPage(1); }, [q, state]);

  const { items, pagination, isLoading, error } = useVMs({ page, pageSize, q, state: state || undefined });

  const { toast } = useToast();

  useEffect(() => {
    const ws = connectEvents((msg) => setWelcome(msg));
    return () => ws.close();
  }, []);

  // Persist role in localStorage (client-only)
  useEffect(() => {
    try {
      const saved = localStorage.getItem("coreRole");
      if (saved === "viewer" || saved === "operator") setRole(saved as Role);
    } catch {}
  }, []);
  useEffect(() => {
    try { localStorage.setItem("coreRole", role); } catch {}
  }, [role]);

  const getStatusColor = (s: string) => {
    switch (s) {
      case "running": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "stopped": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      case "paused": return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };


  const qc = useQueryClient();
  const startMut = useMutation({
    mutationFn: (id: string) => fetchAction(id, "start", role),
    onSuccess: async () => {
      toast({ title: "Start requested", description: "VM start action submitted" });
      await Promise.all([qc.invalidateQueries({queryKey:["vms"]})]);
    },
    onError: (err: any) => {
      toast({ title: "Start failed", description: String(err?.message ?? err), variant: "destructive" as any });
    },
  });
  const stopMut = useMutation({
    mutationFn: (id: string) => fetchAction(id, "stop", role),
    onSuccess: async () => {
      toast({ title: "Stop requested", description: "VM stop action submitted" });
      await Promise.all([qc.invalidateQueries({queryKey:["vms"]})]);
    },
    onError: (err: any) => {
      toast({ title: "Stop failed", description: String(err?.message ?? err), variant: "destructive" as any });
    },
  });
  const restartMut = useMutation({
    mutationFn: (id: string) => fetchAction(id, "restart", role),
    onSuccess: async () => {
      toast({ title: "Restart requested", description: "VM restart action submitted" });
      await Promise.all([qc.invalidateQueries({queryKey:["vms"]})]);
    },
    onError: (err: any) => {
      toast({ title: "Restart failed", description: String(err?.message ?? err), variant: "destructive" as any });
    },
  });
  const pauseMut = useMutation({
    mutationFn: (id: string) => fetchAction(id, "pause", role),
    onSuccess: async () => {
      toast({ title: "Pause requested", description: "VM pause action submitted" });
      await Promise.all([qc.invalidateQueries({queryKey:["vms"]})]);
    },
    onError: (err: any) => {
      toast({ title: "Pause failed", description: String(err?.message ?? err), variant: "destructive" as any });
    },
  });
  const resumeMut = useMutation({
    mutationFn: (id: string) => fetchAction(id, "resume", role),
    onSuccess: async () => {
      toast({ title: "Resume requested", description: "VM resume action submitted" });
      await Promise.all([qc.invalidateQueries({queryKey:["vms"]})]);
    },
    onError: (err: any) => {
      toast({ title: "Resume failed", description: String(err?.message ?? err), variant: "destructive" as any });
    },
  });

  const disabledActions = role !== "operator";

  return (
    <div className="container mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Core VMs</CardTitle>
          <CardDescription>
            {isLoading ? "Loading..." : error ? "Error loading" : `${Array.isArray(items) ? items.length : 0} items`} {" "}
            {welcome ? `— WS welcome: ${JSON.stringify(welcome)}` : null}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3 mb-3 items-center">
            <Input placeholder="Search (q)" value={q} onChange={(e)=>setQ(e.target.value)} className="w-64" />
            <Select value={state} onValueChange={setState}>
              <SelectTrigger className="w-40"><SelectValue placeholder="State" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="">All</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="stopped">Stopped</SelectItem>
                <SelectItem value="paused">Paused</SelectItem>
              </SelectContent>
            </Select>
            {/* Page size */}
            <Select value={String(pageSize)} onValueChange={(v)=>setPageSize(Number(v))}>
              <SelectTrigger className="w-32"><SelectValue placeholder="Page size" /></SelectTrigger>
              <SelectContent>
                {[10,20,50,100].map(n => (<SelectItem key={n} value={String(n)}>{n} / page</SelectItem>))}
              </SelectContent>
            </Select>
            {/* Optional: role switch for actions */}
            <Select value={role} onValueChange={(v)=>setRole(v as Role)}>
              <SelectTrigger className="w-40"><SelectValue placeholder="Role" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="viewer">viewer</SelectItem>
                <SelectItem value="operator">operator</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {error ? (
            <div className="text-sm text-red-500">Failed to load VMs</div>
          ) : !items || (items.length === 0 && !isLoading) ? (
            <div className="text-sm text-muted-foreground">No VMs found.</div>
          ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Node</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Updated</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Array.isArray(items) && items.map((vm: VM) => (
                <TableRow key={vm.id}>
                  <TableCell className="font-mono text-xs">
                    <Link href={`/core/vms/${vm.id}`} aria-label={`View ${vm.name} details`} className="underline underline-offset-4">
                      {vm.id}
                    </Link>
                  </TableCell>
                  <TableCell>
                    <Link href={`/core/vms/${vm.id}`} aria-label={`View ${vm.name} details`} className="hover:underline">
                      {vm.name}
                    </Link>
                  </TableCell>
                  <TableCell><Badge className={getStatusColor(vm.state)}>{vm.state}</Badge></TableCell>
                  <TableCell>{vm.node_id ?? "-"}</TableCell>
                  <TableCell>{vm.created_at}</TableCell>
                  <TableCell>{vm.updated_at}</TableCell>
                  <TableCell className="text-right space-x-2">
                    <Button size="sm" variant="secondary" onClick={()=>startMut.mutate(vm.id)} disabled={disabledActions || startMut.isPending}>Start</Button>
                    <Button size="sm" variant="secondary" onClick={()=>stopMut.mutate(vm.id)} disabled={disabledActions || stopMut.isPending}>Stop</Button>
                    <Button size="sm" variant="secondary" onClick={()=>restartMut.mutate(vm.id)} disabled={disabledActions || restartMut.isPending}>Restart</Button>
                    <Button size="sm" variant="secondary" onClick={()=>pauseMut.mutate(vm.id)} disabled={disabledActions || pauseMut.isPending}>Pause</Button>
                    <Button size="sm" variant="secondary" onClick={()=>resumeMut.mutate(vm.id)} disabled={disabledActions || resumeMut.isPending}>Resume</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          )}

          <div className="flex items-center justify-between mt-3">
            <div className="text-sm text-muted-foreground">
              page={pagination?.page ?? page} pageSize={pagination?.pageSize ?? pageSize} total={pagination?.total ?? "-"} totalPages={pagination?.totalPages ?? "-"}
            </div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={()=>setPage((p)=>Math.max(1, p-1))} disabled={isLoading || (pagination?.page ?? page) <= 1}>Prev</Button>
              <Button size="sm" variant="outline" onClick={()=>setPage((p)=>p+1)} disabled={isLoading || (pagination && pagination.totalPages ? (pagination.page >= pagination.totalPages) : false)}>Next</Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

