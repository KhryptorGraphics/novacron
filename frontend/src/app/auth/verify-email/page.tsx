'use client';

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { Icons } from '@/components/ui/icons';
import { apiService } from '@/lib/api';

export default function VerifyEmailPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const token = searchParams.get('token') ?? '';

  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyState, setVerifyState] = useState<'idle' | 'success' | 'error'>('idle');
  const [verifyError, setVerifyError] = useState('');
  const [email, setEmail] = useState('');
  const [isResending, setIsResending] = useState(false);
  const [isResent, setIsResent] = useState(false);

  const verifyToken = useCallback(async (value: string) => {
    setIsVerifying(true);
    setVerifyState('idle');
    try {
      await apiService.verifyEmail(value);
      setVerifyState('success');
    } catch (error) {
      setVerifyError(error instanceof Error ? error.message : 'This verification link is invalid or has expired.');
      setVerifyState('error');
    } finally {
      setIsVerifying(false);
    }
  }, []);

  useEffect(() => {
    if (token) {
      void verifyToken(token);
    }
  }, [token, verifyToken]);

  async function onResend(event: React.SyntheticEvent) {
    event.preventDefault();
    if (!email) {
      return;
    }

    setIsResending(true);

    try {
      await apiService.resendVerificationEmail(email);

      toast({
        title: 'Success',
        description: 'If an account exists for that email, a verification link has been sent.',
      });

      setIsResent(true);
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to resend the verification email.',
        variant: 'destructive',
      });
    } finally {
      setIsResending(false);
    }
  }

  return (
    <div className="container flex h-screen w-screen flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Email Verification</CardTitle>
          <CardDescription className="text-center">
            {token
              ? 'Verifying your email address'
              : isResent
                ? 'Verification instructions sent'
                : 'Enter your email to receive verification instructions'}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {token ? (
            isVerifying || verifyState === 'idle' ? (
              <div className="text-center py-8">
                <Icons.spinner className="mx-auto h-8 w-8 animate-spin text-muted-foreground" />
                <p className="mt-4 text-sm text-muted-foreground">
                  Verifying your email address...
                </p>
              </div>
            ) : verifyState === 'success' ? (
              <div className="text-center py-8">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                  <Icons.check className="h-8 w-8 text-green-600" />
                </div>
                <p className="mt-4 text-sm text-muted-foreground">
                  Your email has been verified. You can now sign in.
                </p>
                <Button
                  className="w-full mt-6"
                  onClick={() => router.push('/auth/login')}
                >
                  Back to Login
                </Button>
              </div>
            ) : (
              <div className="text-center py-8">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
                  <Icons.mail className="h-8 w-8 text-red-600" />
                </div>
                <p className="mt-4 text-sm text-muted-foreground">
                  {verifyError || 'This verification link is invalid or has expired.'}
                </p>
                <Button
                  className="w-full mt-6"
                  onClick={() => router.push('/auth/verify-email')}
                >
                  Resend Verification Email
                </Button>
              </div>
            )
          ) : isResent ? (
            <div className="text-center py-8">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                <Icons.check className="h-8 w-8 text-green-600" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                If an account exists for <strong>{email}</strong>, a
                verification link has been sent.
              </p>
              <Button
                className="w-full mt-6"
                onClick={() => router.push('/auth/login')}
              >
                Back to Login
              </Button>
            </div>
          ) : (
            <form onSubmit={onResend}>
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  placeholder="user@organization.com"
                  type="email"
                  autoCapitalize="none"
                  autoComplete="email"
                  autoCorrect="off"
                  disabled={isResending}
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <Button className="w-full mt-6" type="submit" disabled={isResending}>
                {isResending && (
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                )}
                Resend Verification Email
              </Button>
            </form>
          )}
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-sm text-muted-foreground text-center">
            Ready to sign in?{' '}
            <Link href="/auth/login" className="hover:text-brand underline underline-offset-4">
              Sign in
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}