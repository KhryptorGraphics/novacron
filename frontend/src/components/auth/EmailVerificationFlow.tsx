"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { SuccessAnimation } from "@/components/ui/success-animation";
import { apiService } from "@/lib/api";
import { cn } from "@/lib/utils";

interface EmailVerificationFlowProps {
  email: string;
  onVerificationComplete?: () => void;
  onSkip?: () => void;
}

export function EmailVerificationFlow({ 
  email, 
  onVerificationComplete, 
  onSkip 
}: EmailVerificationFlowProps) {
  const [step, setStep] = useState<'pending' | 'sent' | 'verified' | 'error'>('pending');
  const [isResending, setIsResending] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Auto-send verification email on mount
  useEffect(() => {
    sendVerificationEmail();
  }, []);

  // Countdown timer for resend button
  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown]);

  const sendVerificationEmail = async () => {
    try {
      setIsResending(true);
      setError(null);
      
      await apiService.resendVerificationEmail(email);
      setStep('sent');
      setCountdown(60); // 60 second cooldown
    } catch (error) {
      console.error('Failed to send verification email:', error);
      setError('Failed to send verification email. Please try again.');
      setStep('error');
    } finally {
      setIsResending(false);
    }
  };

  const handleResend = () => {
    if (countdown === 0) {
      sendVerificationEmail();
    }
  };

  const simulateVerification = () => {
    // For demo purposes - in real app this would be handled by email link
    setTimeout(() => {
      setStep('verified');
      setTimeout(() => {
        onVerificationComplete?.();
      }, 2000);
    }, 2000);
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <CardTitle className="flex items-center justify-center gap-2">
          <Icons.mail className="h-5 w-5 text-blue-600" />
          Email Verification
        </CardTitle>
        <CardDescription>
          We need to verify your email address to secure your account
        </CardDescription>
      </CardHeader>

      <CardContent>
        <AnimatePresence mode="wait">
          {step === 'pending' && (
            <motion.div
              key="pending"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center space-y-4"
            >
              <div className="flex justify-center">
                <Icons.spinner className="h-8 w-8 animate-spin text-blue-600" />
              </div>
              <p className="text-gray-600 dark:text-gray-400">
                Sending verification email...
              </p>
            </motion.div>
          )}

          {step === 'sent' && (
            <motion.div
              key="sent"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="text-center space-y-3">
                <div className="h-16 w-16 mx-auto rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
                  <Icons.mail className="h-8 w-8 text-blue-600" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-semibold">Check your email</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    We've sent a verification link to:
                  </p>
                  <p className="font-medium text-blue-600">{email}</p>
                </div>
              </div>

              <div className="space-y-3">
                <Button
                  onClick={handleResend}
                  disabled={countdown > 0}
                  variant="outline"
                  className="w-full"
                >
                  {countdown > 0 ? (
                    <>
                      <Icons.clock className="mr-2 h-4 w-4" />
                      Resend in {countdown}s
                    </>
                  ) : (
                    <>
                      <Icons.mail className="mr-2 h-4 w-4" />
                      Resend Email
                    </>
                  )}
                </Button>

                {/* Demo button - remove in production */}
                <Button
                  onClick={simulateVerification}
                  variant="default"
                  className="w-full"
                >
                  <Icons.checkCircle2 className="mr-2 h-4 w-4" />
                  Simulate Verification (Demo)
                </Button>

                <Button
                  onClick={onSkip}
                  variant="ghost"
                  className="w-full text-gray-500"
                >
                  Skip for now
                </Button>
              </div>

              <div className="text-xs text-gray-500 text-center space-y-2">
                <p>Didn't receive the email? Check your spam folder.</p>
                <p>The link expires in 24 hours.</p>
              </div>
            </motion.div>
          )}

          {step === 'verified' && (
            <motion.div
              key="verified"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <SuccessAnimation
                title="Email Verified!"
                description="Your email has been successfully verified. You can now access all features."
                onComplete={() => {
                  setTimeout(() => onVerificationComplete?.(), 1000);
                }}
              />
            </motion.div>
          )}

          {step === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center space-y-4"
            >
              <div className="h-16 w-16 mx-auto rounded-full bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
                <Icons.error className="h-8 w-8 text-red-600" />
              </div>
              <div className="space-y-2">
                <h3 className="font-semibold text-red-700 dark:text-red-400">
                  Verification Failed
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {error || "Something went wrong. Please try again."}
                </p>
              </div>
              <Button
                onClick={sendVerificationEmail}
                disabled={isResending}
                className="w-full"
              >
                {isResending ? (
                  <>
                    <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                    Sending...
                  </>
                ) : (
                  'Try Again'
                )}
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}