"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { useToast } from "@/components/ui/use-toast";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { authService } from "@/lib/auth";

interface TwoFactorStatus {
  enabled: boolean;
  backup_codes_remaining: number;
  setup?: boolean;
  setup_at?: string;
  last_used?: string;
}

export default function TwoFactorSettings() {
  const [status, setStatus] = useState<TwoFactorStatus>({ enabled: false, backup_codes_remaining: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [disableCode, setDisableCode] = useState('');
  const [isDisabling, setIsDisabling] = useState(false);
  const [newBackupCodes, setNewBackupCodes] = useState<string[]>([]);
  const [isGeneratingCodes, setIsGeneratingCodes] = useState(false);
  const [showDisableDialog, setShowDisableDialog] = useState(false);
  const [showBackupCodesDialog, setShowBackupCodesDialog] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      setIsLoading(true);
      const statusData = await authService.get2FAStatus();
      setStatus(statusData);
    } catch (error) {
      console.error('Failed to load 2FA status:', error);
      toast({
        title: "Error",
        description: "Failed to load 2FA status. Please refresh the page.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisable2FA = async () => {
    setIsDisabling(true);

    try {
      await authService.disable2FA();

      toast({
        title: "Two-Factor Authentication Disabled",
        description: "Your account is no longer protected by 2FA",
      });

      setStatus({ enabled: false, backup_codes_remaining: 0 });
      setShowDisableDialog(false);
      setDisableCode('');
    } catch (error) {
      console.error('Failed to disable 2FA:', error);
      const errorMessage = error instanceof Error ? error.message : "Failed to disable 2FA. Please try again.";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsDisabling(false);
    }
  };

  const handleGenerateBackupCodes = async () => {
    setIsGeneratingCodes(true);

    try {
      const response = await authService.generateBackupCodes();
      setNewBackupCodes(response.backup_codes);
      setShowBackupCodesDialog(true);

      // Refresh status to update backup codes count
      await loadStatus();

      toast({
        title: "New Backup Codes Generated",
        description: "Your old backup codes are no longer valid. Save the new ones safely.",
      });
    } catch (error) {
      console.error('Failed to generate backup codes:', error);
      toast({
        title: "Error",
        description: "Failed to generate new backup codes. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingCodes(false);
    }
  };

  const downloadBackupCodes = () => {
    const content = `NovaCron Two-Factor Authentication Backup Codes

Generated: ${new Date().toLocaleString()}

These codes can be used to access your account if you lose access to your authenticator app.
Each code can only be used once.

${newBackupCodes.map((code, index) => `${index + 1}. ${code}`).join('\n')}

Keep these codes safe and secure!`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'novacron-2fa-backup-codes.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center py-4">
            <Icons.spinner className="h-6 w-6 animate-spin" />
            <span className="ml-2">Loading 2FA settings...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Icons.shield className="h-5 w-5" />
          Two-Factor Authentication
        </CardTitle>
        <CardDescription>
          Secure your account with an additional layer of protection
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">Status:</span>
              <Badge variant={status.enabled ? "default" : "secondary"}>
                {status.enabled ? "Enabled" : "Disabled"}
              </Badge>
            </div>
            {status.enabled && (
              <p className="text-sm text-gray-600">
                {status.backup_codes_remaining} backup codes remaining
              </p>
            )}
          </div>

          {!status.enabled ? (
            <Button asChild>
              <a href="/auth/setup-2fa">
                <Icons.plus className="mr-2 h-4 w-4" />
                Enable 2FA
              </a>
            </Button>
          ) : (
            <div className="flex gap-2">
              <Button
                onClick={handleGenerateBackupCodes}
                variant="outline"
                disabled={isGeneratingCodes}
              >
                {isGeneratingCodes ? (
                  <>
                    <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Icons.refresh className="mr-2 h-4 w-4" />
                    New Backup Codes
                  </>
                )}
              </Button>

              <Dialog open={showDisableDialog} onOpenChange={setShowDisableDialog}>
                <DialogTrigger asChild>
                  <Button variant="destructive">
                    <Icons.x className="mr-2 h-4 w-4" />
                    Disable 2FA
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Disable Two-Factor Authentication</DialogTitle>
                    <DialogDescription>
                      This will remove the extra security layer from your account.
                      Enter a verification code from your authenticator app to confirm.
                    </DialogDescription>
                  </DialogHeader>

                  <div className="space-y-2">
                    <p className="text-sm text-gray-600">
                      Are you sure you want to disable Two-Factor Authentication?
                      This will remove the extra security layer from your account.
                    </p>
                  </div>

                  <DialogFooter>
                    <Button
                      onClick={() => setShowDisableDialog(false)}
                      variant="outline"
                      disabled={isDisabling}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleDisable2FA}
                      variant="destructive"
                      disabled={isDisabling}
                    >
                      {isDisabling ? (
                        <>
                          <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                          Disabling...
                        </>
                      ) : (
                        "Disable 2FA"
                      )}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          )}
        </div>

        {status.enabled && (
          <div className="pt-4 border-t">
            <h4 className="font-medium mb-2">Security Tips</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Keep your authenticator app updated and backed up</li>
              <li>• Store backup codes in a secure location</li>
              <li>• Generate new backup codes if you think they've been compromised</li>
              <li>• Consider using multiple authenticator apps for redundancy</li>
            </ul>
          </div>
        )}

        {/* Backup Codes Dialog */}
        <Dialog open={showBackupCodesDialog} onOpenChange={setShowBackupCodesDialog}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>New Backup Codes</DialogTitle>
              <DialogDescription>
                Save these codes in a secure location. Each code can only be used once.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-2">
              <div className="grid grid-cols-2 gap-2">
                {newBackupCodes.map((code, index) => (
                  <code
                    key={index}
                    className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-sm text-center"
                  >
                    {code}
                  </code>
                ))}
              </div>
            </div>

            <DialogFooter>
              <Button onClick={downloadBackupCodes} variant="outline">
                <Icons.download className="mr-2 h-4 w-4" />
                Download
              </Button>
              <Button onClick={() => {
                setShowBackupCodesDialog(false);
                setNewBackupCodes([]);
              }}>
                Done
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}