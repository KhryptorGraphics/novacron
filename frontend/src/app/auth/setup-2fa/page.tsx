"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { SuccessAnimation } from "@/components/ui/success-animation";
import { useToast } from "@/components/ui/use-toast";

export default function Setup2FAPage() {
  const [step, setStep] = useState<'setup' | 'verify' | 'complete'>('setup');
  const [isLoading, setIsLoading] = useState(false);
  const [verificationCode, setVerificationCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const { toast } = useToast();

  // Demo QR code and secret - in production this would come from the server
  const qrCodeUrl = "https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=otpauth://totp/NovaCron:user@example.com?secret=JBSWY3DPEHPK3PXP&issuer=NovaCron";
  const backupCodes = [
    "1a2b-3c4d", "5e6f-7g8h", "9i0j-1k2l",
    "3m4n-5o6p", "7q8r-9s0t", "1u2v-3w4x"
  ] || [];

  const handleSkip = () => {
    router.push("/dashboard");
  };

  const handleVerifyCode = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!verificationCode || typeof verificationCode !== 'string' || verificationCode.length !== 6) {
      setError("Please enter a 6-digit verification code");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Simulate API call for verification
      await new Promise(resolve => setTimeout(resolve, 1000));

      // For demo, accept any 6-digit code
      if (verificationCode && verificationCode.length === 6) {
        setStep('complete');
        toast({
          title: "Two-Factor Authentication Enabled",
          description: "Your account is now secured with 2FA",
        });
      } else {
        throw new Error("Invalid code");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Invalid verification code. Please try again.";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      // If a verification API exists, call it here instead of the demo timeout.
      // Example (uncomment and adjust if available):
      // const res = await apiService.verifyTwoFactor(verificationCode);
      // if (!res.success) throw new Error(res.message || "Verification failed");

    }
  };

  const handleComplete = () => {
    router.push("/dashboard");
  };

  return (
    <div className="container relative min-h-screen flex-col items-center justify-center grid lg:max-w-none lg:px-0">
      <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[480px]">
        <div className="flex justify-center mb-4">
          <Link href="/dashboard" className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100">
            <Icons.arrowLeft className="w-4 h-4" />
            Skip to Dashboard
          </Link>
        </div>

        {step === 'setup' && (
          <Card>
            <CardHeader className="text-center">
              <CardTitle className="flex items-center justify-center gap-2">
                <Icons.shield className="h-5 w-5 text-blue-600" />
                Set Up Two-Factor Authentication
              </CardTitle>
              <CardDescription>
                Add an extra layer of security to your account
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="text-center">
                  <h3 className="font-semibold mb-2">1. Install an authenticator app</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    Download Google Authenticator, Authy, or similar app on your phone
                  </p>
                  <div className="flex justify-center gap-4">
                    <div className="flex items-center gap-2 text-sm bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded">
                      <Icons.phone className="h-4 w-4" />
                      Google Authenticator
                    </div>
                    <div className="flex items-center gap-2 text-sm bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded">
                      <Icons.phone className="h-4 w-4" />
                      Authy
                    </div>
                  </div>
                </div>

                <div className="text-center space-y-4">
                  <h3 className="font-semibold">2. Scan the QR code</h3>
                  <div className="flex justify-center">
                    <div className="p-4 bg-white rounded-lg shadow-sm border">
                      <img
                        src={qrCodeUrl}
                        alt="QR Code for 2FA setup"
                        className="w-48 h-48"
                      />
                    </div>
                  </div>
                  <p className="text-xs text-gray-500">
                    Or manually enter this code: <code className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">JBSWY3DPEHPK3PXP</code>
                  </p>
                </div>

                <div className="text-center">
                  <h3 className="font-semibold mb-2">3. Save your backup codes</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                    Store these codes safely. You can use them to access your account if you lose your phone.
                  </p>
                  <div className="grid grid-cols-2 gap-2 max-w-xs mx-auto">
                    {Array.isArray(backupCodes) && backupCodes.length > 0 ? backupCodes.map((code, index) => (
                      <code key={index} className="bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-sm">
                        {code || ''}
                      </code>
                    )) : (
                      <div className="col-span-2 text-center text-muted-foreground text-sm">
                        No backup codes available
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={() => setStep('verify')}
                  className="flex-1"
                >
                  <Icons.arrowRight className="mr-2 h-4 w-4" />
                  Continue to Verification
                </Button>
                <Button
                  onClick={handleSkip}
                  variant="outline"
                >
                  Skip
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {step === 'verify' && (
          <Card>
            <CardHeader className="text-center">
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
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={6}
                    placeholder="123456"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, ''))}
                    className={error ? "border-red-500" : ""}
                    autoComplete="one-time-code"
                  />
                  {error && (
                    <p className="text-sm text-red-600">{error}</p>
                  )}
                </div>

                <div className="flex gap-2">
                  <Button
                    type="submit"
                    disabled={isLoading || verificationCode.length !== 6}
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
                        Verify & Enable
                      </>
                    )}
                  </Button>
                  <Button
                    type="button"
                    onClick={() => setStep('setup')}
                    variant="outline"
                  >
                    Back
                  </Button>
                </div>
              </form>

              <div className="mt-4 text-center">
                <Button
                  onClick={handleSkip}
                  variant="ghost"
                  className="text-gray-500"
                >
                  Skip for now
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {step === 'complete' && (
          <Card>
            <CardContent className="pt-6">
              <SuccessAnimation
                title="Two-Factor Authentication Enabled!"
                description="Your account is now protected with an additional layer of security. You'll be asked for a verification code when signing in."
                onComplete={handleComplete}
              />
              <div className="text-center mt-6">
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