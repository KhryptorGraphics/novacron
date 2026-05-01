"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ArrowRight, CheckCircle, AlertCircle } from "lucide-react";
import { initiateLiveMigration } from "@/lib/api/client";

interface VMMigrationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  vmId: string | null;
}

export function VMMigrationDialog({ open, onOpenChange, vmId }: VMMigrationDialogProps) {
  const [targetNode, setTargetNode] = useState("");
  const [priority, setPriority] = useState<"low" | "normal" | "high">("normal");
  const [bandwidth, setBandwidth] = useState("");
  const [maxDowntime, setMaxDowntime] = useState("30");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<{ migrationId: string; status: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setTargetNode("");
    setPriority("normal");
    setBandwidth("");
    setMaxDowntime("30");
    setResult(null);
    setError(null);
  };

  const submitMigration = async () => {
    if (!vmId) {
      setError("VM ID is required.");
      return;
    }
    if (!targetNode.trim()) {
      setError("Target node is required.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const response = await initiateLiveMigration(vmId, targetNode.trim(), {
        priority,
        ...(bandwidth ? { bandwidth: Number(bandwidth) } : {}),
        ...(maxDowntime ? { maxDowntime: Number(maxDowntime) } : {}),
      });
      if (response.error) {
        throw new Error(response.error.message);
      }
      setResult(response.data || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Migration request failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        onOpenChange(nextOpen);
        if (!nextOpen) reset();
      }}
    >
      <DialogContent className="sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ArrowRight className="h-5 w-5" />
            Migrate Virtual Machine
          </DialogTitle>
          <DialogDescription>
            Submit a canonical live migration request for {vmId || "the selected VM"}.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="grid gap-2">
            <Label htmlFor="target-node">Target node</Label>
            <Input
              id="target-node"
              value={targetNode}
              onChange={(event) => setTargetNode(event.target.value)}
              placeholder="node-02"
            />
          </div>

          <div className="grid gap-2">
            <Label>Priority</Label>
            <Select value={priority} onValueChange={(value) => setPriority(value as "low" | "normal" | "high")}>
              <SelectTrigger>
                <SelectValue placeholder="Priority" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="normal">Normal</SelectItem>
                <SelectItem value="high">High</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="grid gap-2">
              <Label htmlFor="max-downtime">Max downtime ms</Label>
              <Input
                id="max-downtime"
                type="number"
                min={0}
                value={maxDowntime}
                onChange={(event) => setMaxDowntime(event.target.value)}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="bandwidth">Bandwidth Mbps</Label>
              <Input
                id="bandwidth"
                type="number"
                min={0}
                value={bandwidth}
                onChange={(event) => setBandwidth(event.target.value)}
                placeholder="auto"
              />
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Migration request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {result && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertTitle>Migration queued</AlertTitle>
              <AlertDescription>
                {result.migrationId} is {result.status}.
              </AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button onClick={submitMigration} disabled={isSubmitting || !vmId}>
            {isSubmitting ? "Submitting..." : "Start Migration"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
