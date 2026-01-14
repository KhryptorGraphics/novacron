"use client";

import { postVMAction } from "@/lib/api/vms";

export async function fetchAction(id: string, action: "start" | "stop" | "restart" | "pause" | "resume", role: "viewer"|"operator") {
  return postVMAction(id, action, { role });
}

