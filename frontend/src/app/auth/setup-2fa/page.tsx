"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { authService } from "@/lib/auth";

export default function Setup2FAPage() {
  const [step, setStep] = useState<'setup' | 'verify' | 'complete'>('setup');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSetup, setLoadingSetup] = useState(true);
  const [verificationCode, setVerificationCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [qrCodeUrl, setQrCodeUrl] = useState<string>('');
  const [secret, setSecret] = useState<string>('');
  const [backupCodes, setBackupCodes] = useState<string[]>([]);
  const router = useRouter();
  const { toast } = useToast();

  // Load 2FA setup data when component mounts
  useEffect(() => {
    const initSetup = async () => {
      try {
        // Check if user is authenticated
        if (!authService.isAuthenticated()) {
          router.push('/auth/login');
          return;
        }

        setLoadingSetup(true);
        const setupData = await authService.setup2FA();
        setQrCodeUrl(setupData.qr_code);
        setSecret(setupData.secret);
        setBackupCodes(setupData.backup_codes);
      } catch (error) {
        console.error('Failed to initialize 2FA setup:', error);
        toast({
          title: "Error",
          description: "Failed to initialize 2FA setup. Please try again.",
          variant: "destructive",
        });
        // Fall back to demo data if API fails
        setQrCodeUrl("https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=otpauth://totp/NovaCron:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=NovaCron");
        setSecret("JBSWY3DPEHPK3PXP");
        setBackupCodes([
          "1a2b-3c4d", "5e6f-7g8h", "9i0j-1k2l",
          "3m4n-5o6p", "7q8r-9s0t", "1u2v-3w4x"
        ]);
      } finally {
        setLoadingSetup(false);
      }
    };

    initSetup();
  }, [router, toast]);

  const handleSkip = () => {
    router.push("/dashboard");
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!verificationCode || verificationCode.length !== 6) {
      setError("Please enter a 6-digit verification code");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const user = authService.getCurrentUser();
      if (!user) {
        throw new Error("User not authenticated");
      }

      const verifyResponse = await authService.verify2FASetup({
        user_id: user.id,
        code: verificationCode,
      });

      if (verifyResponse.valid) {
        await authService.enable2FA(verificationCode);
        setStep('complete');
        toast({
          title: "Two-Factor Authentication Enabled",
          description: "Your account is now secured with 2FA",
        });
      } else {
        throw new Error(verifyResponse.error || "Verification failed");
      }
    } catch (error) {
      console.error('2FA verification failed:', error);
      const errorMessage = error instanceof Error ? error.message : "Invalid verification code. Please try again.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleComplete = () => {
    router.push("/dashboard");
  };

  return (
    <div className="container relative min-h-screen flex-col items-center justify-center grid lg:max-w-none lg:px-0">
      <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[480px]">
        <div className="flex justify-center mb-4">
          <Link href="/dashboard" className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900">
            ← Skip to Dashboard
          </Link>
        </div>

        {step === 'setup' && (
          <Card>
            <CardHeader>
              <CardTitle>Set Up Two-Factor Authentication</CardTitle>
              <CardDescription>
                Scan the QR code with your authenticator app
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {loadingSetup ? (
                <div className="flex justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                </div>
              ) : (
                <>
                  <div className="flex justify-center">
                    <img src={qrCodeUrl} alt="2FA QR Code" className="w-48 h-48" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-600 mb-2">Or enter this code manually:</p>
                    <code className="bg-gray-100 px-3 py-1 rounded">{secret}</code>
                  </div>
                  <div className="space-y-2">
                    <Label>Backup Codes</Label>
                    <div className="grid grid-cols-2 gap-2">
                      {backupCodes.map((code, index) => (
                        <code key={index} className="bg-gray-100 px-2 py-1 rounded text-sm">
                          {code}
                        </code>
                      ))}
                    </div>
                  </div>
                </>
              )}
              <div className="flex gap-2">
                <Button onClick={() => setStep('verify')} className="flex-1" disabled={loadingSetup}>
                  Continue to Verification
                </Button>
                <Button onClick={handleSkip} variant="outline" disabled={loadingSetup}>
                  Skip
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {step === 'verify' && (
          <Card>
            <CardHeader>
              <CardTitle>Verify Your Setup</CardTitle>
              <CardDescription>
                Enter the 6-digit code from your authenticator app
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleVerifyCode} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="code">Verification Code</Label>
                  <Input
                    id="code"
                    type="text"
                    placeholder="000000"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value)}
                    maxLength={6}
                    disabled={isLoading}
                  />
                  {error && <p className="text-sm text-red-600">{error}</p>}
                </div>
                <Button type="submit" className="w-full" disabled={isLoading}>
                  {isLoading ? "Verifying..." : "Verify and Enable 2FA"}
                </Button>
              </form>
            </CardContent>
          </Card>
        )}

        {step === 'complete' && (
          <Card>
            <CardHeader>
              <CardTitle>Setup Complete!</CardTitle>
              <CardDescription>
                Two-factor authentication has been enabled for your account
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center space-y-4">
                <div className="text-green-600 text-6xl">✓</div>
                <p>Your account is now more secure</p>
                <Button onClick={handleComplete} className="w-full">
                  Continue to Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

