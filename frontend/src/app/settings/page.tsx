"use client";

export const dynamic = "force-dynamic";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import AuthGuard from "@/components/auth/AuthGuard";
import TwoFactorSettings from "@/components/auth/TwoFactorSettings";
import { useAuth } from "@/hooks/useAuth";
import { KeyRound, Settings, Shield, UserCircle2 } from "lucide-react";

export default function SettingsPage() {
  const { user } = useAuth();
  const roles = user?.roles?.length ? user.roles : user?.role ? [user.role] : [];

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-bold">Settings</h1>
            <p className="text-muted-foreground">
              Account and security preferences only. Global system configuration is out of scope for the release candidate.
            </p>
          </div>
          <Badge variant="outline">Account & Security Only</Badge>
        </div>

        <Alert>
          <Settings className="h-4 w-4" />
          <AlertTitle>Scoped release surface</AlertTitle>
          <AlertDescription>
            Notification, backup, API, and infrastructure settings were removed from this routed page because they do not have a live canonical backend.
          </AlertDescription>
        </Alert>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserCircle2 className="h-5 w-5" />
                Account
              </CardTitle>
              <CardDescription>Identity information derived from the current canonical auth token.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Email</div>
                <div className="mt-1 font-medium">{user?.email || 'Unknown user'}</div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Name</div>
                <div className="mt-1 font-medium">
                  {[user?.firstName, user?.lastName].filter(Boolean).join(" ") || "Not provided"}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Roles</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {roles.length > 0 ? roles.map((role) => (
                    <Badge key={role} variant="secondary">{role}</Badge>
                  )) : <span className="text-sm text-muted-foreground">No roles present in the session token.</span>}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm font-medium text-muted-foreground">Tenant</div>
                <div className="mt-1 font-medium">{user?.tenantId || 'default'}</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Scope
              </CardTitle>
              <CardDescription>Release-candidate settings are limited to authentication and session security.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <div className="rounded-lg border p-4">
                <div className="flex items-center gap-2 font-medium text-foreground">
                  <KeyRound className="h-4 w-4" />
                  Password reset
                </div>
                <p className="mt-2">Password reset is live through the canonical auth surface and available from the login flow.</p>
              </div>
              <div className="rounded-lg border p-4">
                <div className="font-medium text-foreground">Email verification</div>
                <p className="mt-2">Email verification stays deferred for the release candidate and is intentionally not managed from this page.</p>
              </div>
              <div className="rounded-lg border p-4">
                <div className="font-medium text-foreground">Global system configuration</div>
                <p className="mt-2">Infrastructure-wide settings belong to deferred admin features and are not exposed here.</p>
              </div>
            </CardContent>
          </Card>
        </div>

        <TwoFactorSettings />
      </div>
    </AuthGuard>
  );
}
