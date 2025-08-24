"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { RegistrationWizard } from "@/components/auth/RegistrationWizard";
import { RegistrationData } from "@/lib/validation";
import { apiService } from "@/lib/api";
import { useToast } from "@/components/ui/use-toast";

export default function RegisterPage() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const { toast } = useToast();

  async function onComplete(data: RegistrationData) {
    setIsLoading(true);
    
    try {
      // Call the API with the full registration data
      await apiService.register({
        firstName: data.firstName,
        lastName: data.lastName,
        email: data.email,
        password: data.password,
        accountType: data.accountType,
        organizationName: data.organizationName,
        organizationSize: data.organizationSize,
        phone: data.phone,
        enableTwoFactor: data.enableTwoFactor,
      });
      
      // Success is handled by the wizard component
      // (shows verification flow, then success animation, then redirects)
    } catch (error) {
      console.error("Registration error:", error);
      toast({
        title: "Registration Failed",
        description: "Unable to create your account. Please try again.",
        variant: "destructive",
      });
      throw error; // Re-throw to be handled by the wizard
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="container relative min-h-screen flex-col items-center justify-center grid lg:max-w-none lg:grid-cols-2 lg:px-0">
      <div className="relative hidden h-full flex-col bg-muted p-10 text-white dark:border-r lg:flex">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-900 to-black" />
        <div className="relative z-20 flex items-center text-lg font-medium">
          <div className="mr-2 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center">
            <span className="text-xl font-bold">N</span>
          </div>
          NovaCron
        </div>
        <div className="relative z-20 mt-auto space-y-6">
          <div className="space-y-4">
            <h2 className="text-3xl font-bold">Enterprise-Grade VM Management</h2>
            <p className="text-lg text-gray-300">
              Join thousands of organizations using NovaCron to manage their distributed infrastructure
            </p>
          </div>
          
          <div className="grid gap-4">
            <div className="flex items-start gap-3">
              <div className="mt-1 h-5 w-5 rounded-full bg-blue-500/20 flex items-center justify-center">
                <span className="text-xs text-blue-400">✓</span>
              </div>
              <div>
                <h3 className="font-semibold">Automated VM Lifecycle</h3>
                <p className="text-sm text-gray-400">Provision, monitor, and manage VMs at scale</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="mt-1 h-5 w-5 rounded-full bg-blue-500/20 flex items-center justify-center">
                <span className="text-xs text-blue-400">✓</span>
              </div>
              <div>
                <h3 className="font-semibold">Live Migration</h3>
                <p className="text-sm text-gray-400">Zero-downtime migrations with WAN optimization</p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <div className="mt-1 h-5 w-5 rounded-full bg-blue-500/20 flex items-center justify-center">
                <span className="text-xs text-blue-400">✓</span>
              </div>
              <div>
                <h3 className="font-semibold">Advanced Monitoring</h3>
                <p className="text-sm text-gray-400">Real-time metrics with predictive analytics</p>
              </div>
            </div>
          </div>
          
          <blockquote className="border-l-2 border-blue-500 pl-4 space-y-2">
            <p className="text-lg italic">
              "NovaCron has revolutionized how we manage our distributed infrastructure. The automation capabilities are unmatched."
            </p>
            <footer className="text-sm text-gray-400">— Sofia Davis, CTO at TechCorp</footer>
          </blockquote>
        </div>
      </div>
      
      <div className="lg:p-8">
        <div className="mx-auto flex w-full flex-col justify-center space-y-6">
          <div className="flex justify-center mb-4">
            <Link href="/" className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Home
            </Link>
          </div>
          
          <RegistrationWizard onComplete={onComplete} />
          
          <p className="text-center text-sm text-muted-foreground mt-4">
            Already have an account?{" "}
            <Link
              href="/auth/login"
              className="font-medium text-blue-600 hover:text-blue-500 underline underline-offset-4"
            >
              Sign in instead
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}