"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { Icons } from "@/components/ui/icons";
import Link from "next/link";
import { apiService } from "@/lib/api";

export default function ForgotPasswordPage() {
  const router = useRouter();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);

  async function onSubmit(event: React.SyntheticEvent) {
    event.preventDefault();
    setIsLoading(true);

    try {
      await apiService.forgotPassword({ email });
      
      toast({
        title: "Success",
        description: "Password reset email sent. Please check your inbox.",
      });
      
      setIsSubmitted(true);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send password reset email. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="container flex h-screen w-screen flex-col items-center justify-center">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Forgot Password</CardTitle>
          <CardDescription className="text-center">
            {isSubmitted
              ? "Please check your email for password reset instructions"
              : "Enter your email to receive password reset instructions"}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          {isSubmitted ? (
            <div className="text-center py-8">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                <Icons.check className="h-8 w-8 text-green-600" />
              </div>
              <p className="mt-4 text-sm text-muted-foreground">
                We've sent password reset instructions to <strong>{email}</strong>
              </p>
              <Button 
                className="w-full mt-6" 
                onClick={() => router.push("/auth/login")}
              >
                Back to Login
              </Button>
            </div>
          ) : (
            <form onSubmit={onSubmit}>
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  placeholder="name@example.com"
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
              <Button className="w-full mt-6" type="submit" disabled={isLoading}>
                {isLoading && (
                  <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                )}
                Send Reset Instructions
              </Button>
            </form>
          )}
        </CardContent>
        <CardFooter className="flex flex-col">
          <div className="text-sm text-muted-foreground text-center">
            Remember your password?{" "}
            <Link href="/auth/login" className="hover:text-brand underline underline-offset-4">
              Sign in
            </Link>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}