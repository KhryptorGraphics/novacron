"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { useToast } from "@/components/ui/use-toast";
import { authService } from "@/lib/auth";

interface TwoFactorLoginProps {
  tempToken: string;
  onSuccess: (token: string) => void;
  onCancel: () => void;
}

export default function TwoFactorLogin({ tempToken, onSuccess, onCancel }: TwoFactorLoginProps) {
  const [code, setCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useBackupCode, setUseBackupCode] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!code || code.length < 6) {
      setError(useBackupCode ? "Please enter your backup code" : "Please enter the 6-digit code from your authenticator app");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // For login flow, we might not have a current user yet, so we'll need to handle this differently
      // The backend should be able to identify the user from the temp_token
      const response = await authService.verify2FALogin({
        user_id: "", // Empty for login flow, backend should derive from temp_token
        code: code,
        is_backup_code: useBackupCode,
        temp_token: tempToken,
      });

      if (response.token) {
        // Store the real token
        authService.setToken(response.token, response.user);
        authService.removeTempToken();

        toast({
          title: "Login Successful",
          description: "You have been authenticated successfully.",
        });

        onSuccess(response.token);
      } else {
        throw new Error("Authentication failed");
      }
    } catch (error) {
      console.error('2FA login verification failed:', error);
      const errorMessage = error instanceof Error ? error.message : "Invalid code. Please try again.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    authService.removeTempToken();
    onCancel();
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="text-center">
        <CardTitle className="flex items-center justify-center gap-2">
          <Icons.shield className="h-5 w-5 text-blue-600" />
          Two-Factor Authentication
        </CardTitle>
        <CardDescription>
          {useBackupCode
            ? "Enter one of your backup codes to continue"
            : "Enter the 6-digit code from your authenticator app"
          }
        </CardDescription>
      </CardHeader>

      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="code">
              {useBackupCode ? "Backup Code" : "Verification Code"}
            </Label>
            <Input
              id="code"
              type="text"
              inputMode={useBackupCode ? "text" : "numeric"}
              pattern={useBackupCode ? undefined : "[0-9]*"}
              maxLength={useBackupCode ? 12 : 6}
              placeholder={useBackupCode ? "xxxx-xxxx" : "123456"}
              value={code}
              onChange={(e) => {
                const value = useBackupCode ? e.target.value : e.target.value.replace(/\D/g, '');
                setCode(value);
              }}
              className={error ? "border-red-500" : ""}
              autoComplete="one-time-code"
              disabled={isLoading}
            />
            {error && (
              <p className="text-sm text-red-600">{error}</p>
            )}
          </div>

          <div className="flex gap-2">
            <Button
              type="submit"
              disabled={isLoading || (!useBackupCode && code.length !== 6) || (useBackupCode && code.length === 0)}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                  Verifying...
                </>
              ) : (
                <>
                  <Icons.checkCircle2 className="mr-2 h-4 w-4" />
                  Verify
                </>
              )}
            </Button>
            <Button
              type="button"
              onClick={handleCancel}
              variant="outline"
              disabled={isLoading}
            >
              Cancel
            </Button>
          </div>

          <div className="text-center space-y-2">
            <Button
              type="button"
              onClick={() => {
                setUseBackupCode(!useBackupCode);
                setCode('');
                setError(null);
              }}
              variant="ghost"
              className="text-sm text-gray-500"
              disabled={isLoading}
            >
              {useBackupCode ? "Use authenticator code instead" : "Use backup code"}
            </Button>

            <div className="text-xs text-gray-500">
              <p>Can't access your authenticator app?</p>
              <p>Use one of your backup codes to sign in.</p>
            </div>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}