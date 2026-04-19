"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { Icons } from "@/components/ui/icons";
import Link from "next/link";
import { useAuth } from "@/hooks/useAuth";
import TwoFactorLogin from "@/components/auth/TwoFactorLogin";
import { authService } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isGitHubLoading, setIsGitHubLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const { login, verify2FA, requires2FA, tempToken } = useAuth();

  async function onSubmit(event: React.SyntheticEvent) {
    event.preventDefault();
    setIsLoading(true);

    try {
      await login(email, password);

      if (!requires2FA) {
        toast({
          title: "Success",
          description: "You have been logged in successfully.",
        });
        router.push("/dashboard");
      }
    } catch (error) {
      console.error('Login error:', error);
      toast({
        title: "Error",
        description: "Invalid email or password. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }

  const handle2FASuccess = (token: string) => {
    toast({
      title: "Success",
      description: "You have been logged in successfully.",
    });
    router.push("/dashboard");
  };

  const handle2FACancel = () => {
    // Reset form state to allow new login attempt
    setEmail("");
    setPassword("");
  };

  async function onGitHubLogin() {
    setIsGitHubLoading(true);

    try {
      const response = await authService.getGitHubAuthorizationUrl('/dashboard');
      window.location.assign(response.authorizationUrl);
    } catch (error) {
      console.error('GitHub login error:', error);
      toast({
        title: "GitHub Login Unavailable",
        description: "GitHub OAuth is not configured or could not be started.",
        variant: "destructive",
      });
      setIsGitHubLoading(false);
    }
  }

  if (requires2FA && tempToken) {
    return (
      <div className="container flex h-screen w-screen flex-col items-center justify-center">
        <TwoFactorLogin
          tempToken={tempToken}
          onSuccess={handle2FASuccess}
          onCancel={handle2FACancel}
        />
      </div>
    );
  }

  return (
    <div className="container flex h-screen w-screen flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Login</CardTitle>
          <CardDescription className="text-center">
            Enter your email and password to access your account
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <Button
            className="w-full"
            type="button"
            variant="outline"
            onClick={onGitHubLogin}
            disabled={isLoading || isGitHubLoading}
          >
            {(isGitHubLoading || isLoading) && (
              <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
            )}
            {!(isGitHubLoading || isLoading) && (
              <svg
                aria-hidden="true"
                className="mr-2 h-4 w-4"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 .5C5.65.5.5 5.66.5 12.02c0 5.09 3.29 9.4 7.86 10.92.58.11.79-.25.79-.56 0-.28-.01-1.19-.02-2.16-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.27-1.68-1.27-1.68-1.04-.71.08-.69.08-.69 1.15.08 1.75 1.19 1.75 1.19 1.02 1.76 2.68 1.25 3.33.96.1-.74.4-1.25.73-1.53-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.47.11-3.07 0 0 .96-.31 3.15 1.18a10.9 10.9 0 0 1 5.73 0c2.19-1.49 3.15-1.18 3.15-1.18.62 1.6.23 2.78.11 3.07.74.81 1.18 1.84 1.18 3.1 0 4.42-2.69 5.39-5.25 5.67.41.35.77 1.04.77 2.09 0 1.51-.01 2.73-.01 3.1 0 .31.21.68.8.56A11.53 11.53 0 0 0 23.5 12.02C23.5 5.66 18.35.5 12 .5Z" />
              </svg>
            )}
            Continue with GitHub
          </Button>
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">Or continue with email</span>
            </div>
          </div>
          <form onSubmit={onSubmit}>
            <div className="grid gap-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                placeholder="user@organization.com"
                type="email"
                autoCapitalize="none"
                autoComplete="email"
                autoCorrect="off"
                disabled={isLoading}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="grid gap-2 mt-4">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                placeholder="••••••••"
                type="password"
                autoComplete="current-password"
                disabled={isLoading}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button className="w-full mt-6" type="submit" disabled={isLoading}>
              {isLoading && (
                <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
              )}
              Sign In
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-sm text-muted-foreground text-center">
            <Link href="/auth/forgot-password" className="hover:text-brand underline underline-offset-4">
              Forgot your password?
            </Link>
          </div>
          <div className="text-sm text-muted-foreground text-center mt-2">
            Don't have an account?{" "}
            <Link href="/auth/register" className="hover:text-brand underline underline-offset-4">
              Sign up
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
