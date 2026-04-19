"use client";

export const dynamic = 'force-dynamic';

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { authService } from "@/lib/auth";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Icons } from "@/components/ui/icons";
import { useToast } from "@/components/ui/use-toast";

function decodeFragmentPayload<T>(value: string | null): T | null {
  if (!value) {
    return null;
  }

  try {
    const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
    const padded = normalized + '==='.slice((normalized.length + 3) % 4);
    const decoded = window.atob(padded);
    return JSON.parse(decoded) as T;
  } catch (error) {
    console.error('Failed to decode OAuth callback payload:', error);
    return null;
  }
}

export default function GitHubCallbackPage() {
  const router = useRouter();
  const { toast } = useToast();

  useEffect(() => {
    const fragment = window.location.hash.startsWith('#')
      ? window.location.hash.slice(1)
      : window.location.hash;
    const params = new URLSearchParams(fragment);
    const token = params.get('token');
    const encodedUser = params.get('user');
    const redirectTo = params.get('redirect_to') || '/dashboard';
    const user = decodeFragmentPayload(encodedUser);

    if (!token || !user) {
      toast({
        title: "GitHub Login Failed",
        description: "The GitHub callback did not include a valid session.",
        variant: "destructive",
      });
      router.replace('/auth/login');
      return;
    }

    authService.setToken(token, user);
    router.replace(redirectTo);
  }, [router, toast]);

  return (
    <div className="container flex h-screen w-screen flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Finishing GitHub Sign In</CardTitle>
          <CardDescription className="text-center">
            NovaCron is exchanging your GitHub session and admitting you into the local cluster.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-8">
          <Icons.spinner className="h-6 w-6 animate-spin" />
        </CardContent>
      </Card>
    </div>
  );
}
