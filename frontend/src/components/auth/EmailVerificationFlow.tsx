"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";

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
  void onVerificationComplete;

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
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div
            role="status"
            aria-live="polite"
            className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-100"
          >
            Email verification is not available on the canonical release-candidate server yet.
          </div>

          <div className="text-center space-y-3">
            <div className="h-16 w-16 mx-auto rounded-full bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
              <Icons.mail className="h-8 w-8 text-blue-600" />
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold">Verification Pending</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                The routed signup flow no longer depends on email verification.
              </p>
              <p className="font-medium text-blue-600 break-all">{email}</p>
            </div>
          </div>

          <div className="space-y-3">
            <Button
              disabled
              variant="outline"
              className="w-full"
              aria-disabled="true"
            >
              <Icons.mail className="mr-2 h-4 w-4" />
              Resend Email Unavailable
            </Button>

            <Button
              onClick={onSkip}
              variant="default"
              className="w-full"
            >
              Continue Without Verification
            </Button>
          </div>

          <div className="text-xs text-center text-gray-500 dark:text-gray-400">
            Verification and resend endpoints remain deferred until the canonical backend supports them.
          </div>
        </motion.div>
      </CardContent>
    </Card>
  );
}
