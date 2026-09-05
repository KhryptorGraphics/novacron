'use client';

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/components/ui/use-toast';
import { Icons } from '@/components/ui/icons';
import { PasswordStrengthIndicator } from '@/components/auth/PasswordStrengthIndicator';
import { authService } from '@/lib/auth';

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const token = searchParams.get('token') ?? '';

  const [isLoading, setIsLoading] = useState(false);
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

  async function onSubmit(event: React.SyntheticEvent) {
    event.preventDefault();

    if (password !== confirm) {
      toast({
        title: 'Error',
        description: 'Passwords do not match',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);

    try {
      await authService.resetPassword({ token, password });

      toast({
        title: 'Success',
        description: 'Your password has been reset. You can now sign in.',
      });

      setIsSubmitted(true);
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to reset password. The link may have expired.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="container flex h-screen w-screen flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Reset Password</CardTitle>
          <CardDescription className="text-center">
            {isSubmitted
              ? 'Your password has been reset successfully'
              : 'Choose a new password for your account'}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {!token ? (
            <div className="text-center py-8">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
                <Icons.lock className="h-8 w-8 text-red-600" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                This reset link is invalid. Request a new one from the forgot
                password page.
              </p>
              <Button
                className="w-full mt-6"
                onClick={() => router.push('/auth/forgot-password')}
              >
                Back to Forgot Password
              </Button>
            </div>
          ) : isSubmitted ? (
            <div className="text-center py-8">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                <Icons.check className="h-8 w-8 text-green-600" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                Your password has been reset. Sign in with your new password.
              </p>
              <Button
                className="w-full mt-6"
                onClick={() => router.push('/auth/login')}
              >
                Back to Login
              </Button>
            </div>
          ) : (
            <form onSubmit={onSubmit}>
              <div className="grid gap-2">
                <Label htmlFor="password">New Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  disabled={isLoading}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
              <div className="mt-4">
                <PasswordStrengthIndicator password={password} />
              </div>
              <div className="grid gap-2 mt-4">
                <Label htmlFor="confirm">Confirm Password</Label>
                <Input
                  id="confirm"
                  type="password"
                  autoComplete="new-password"
                  disabled={isLoading}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                />
              </div>
              <Button className="w-full mt-6" type="submit" disabled={isLoading}>
                {isLoading && (
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                )}
                Reset Password
              </Button>
            </form>
          )}
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-sm text-muted-foreground text-center">
            Remember your password?{' '}
            <Link href="/auth/login" className="hover:text-brand underline underline-offset-4">
              Sign in
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}