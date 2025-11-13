"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { PasswordInput } from "@/components/ui/password-input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { Icons } from "@/components/ui/icons";
import { PasswordStrengthIndicator } from "./PasswordStrengthIndicator";
import { EmailVerificationFlow } from "./EmailVerificationFlow";
import { SuccessAnimation } from "@/components/ui/success-animation";
import {
  validateRegistrationStep,
  validateEmail,
  debounce,
  RegistrationData
} from "@/lib/validation";
import { apiService } from "@/lib/api";
import { cn } from "@/lib/utils";

interface RegistrationWizardProps {
  onComplete?: (data: RegistrationData) => Promise<void>;
}

export function RegistrationWizard({ onComplete }: RegistrationWizardProps) {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [emailAvailable, setEmailAvailable] = useState<boolean | null>(null);
  const [checkingEmail, setCheckingEmail] = useState(false);
  const [registrationFlow, setRegistrationFlow] = useState<'registration' | 'verification' | 'success'>('registration');
  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  
  const [formData, setFormData] = useState<RegistrationData>({
    accountType: '',
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    organizationName: '',
    organizationSize: '',
    phone: '',
    acceptTerms: false,
    enableTwoFactor: false
  });
  
  const totalSteps = formData.accountType === 'organization' ? 4 : 3;
  const progress = (currentStep / totalSteps) * 100;
  
  // Step configuration
  const getStepInfo = () => {
    const steps = [
      { number: 1, title: "Account Type", description: "Choose your account type" },
      { number: 2, title: "Personal Info", description: "Tell us about yourself" },
    ];
    
    if (formData.accountType === 'organization') {
      steps.push({ number: 3, title: "Organization", description: "Organization details" });
    }
    
    steps.push({
      number: steps.length + 1,
      title: "Security",
      description: "Secure your account"
    });
    
    return steps;
  };
  
  // Email availability check
  const checkEmailAvailability = debounce(async (email: string) => {
    if (!email || !validateEmail(email).isValid) {
      setEmailAvailable(null);
      return;
    }
    
    setCheckingEmail(true);
    try {
      // Use actual API call
      const result = await apiService.checkEmailAvailability(email);
      setEmailAvailable(result.available);
    } catch (error) {
      console.error("Error checking email:", error);
      // Fallback for demo - check if email contains "taken"
      setEmailAvailable(!email.includes("taken"));
    } finally {
      setCheckingEmail(false);
    }
  }, 500);
  
  useEffect(() => {
    if (formData.email) {
      checkEmailAvailability(formData.email);
    }
  }, [formData.email]);
  
  const handleNext = async () => {
    const validation = validateRegistrationStep(currentStep, formData);
    
    if (!validation.isValid) {
      setErrors(validation.errors);
      return;
    }
    
    setErrors([]);
    
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1);
    } else {
      // Final step - submit registration
      setIsLoading(true);
      try {
        if (onComplete) {
          await onComplete(formData);
        } else {
          // Use actual API service
          const result = await apiService.register({
            firstName: formData.firstName,
            lastName: formData.lastName,
            email: formData.email,
            password: formData.password,
            accountType: formData.accountType,
            organizationName: formData.organizationName,
            organizationSize: formData.organizationSize,
            phone: formData.phone,
            enableTwoFactor: formData.enableTwoFactor,
          });

          if (result.success) {
            setRegistrationSuccess(true);
            // Move to email verification flow
            setRegistrationFlow('verification');
          } else {
            throw new Error('Registration failed');
          }
        }
      } catch (error) {
        console.error("Registration error:", error);
        setErrors(["Registration failed. Please try again."]);
      } finally {
        setIsLoading(false);
      }
    }
  };
  
  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
      setErrors([]);
    }
  };
  
  const updateFormData = (field: keyof RegistrationData, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear errors when user makes changes
    if (errors.length > 0) {
      setErrors([]);
    }
  };

  const handleVerificationComplete = () => {
    setRegistrationFlow('success');
    setTimeout(() => {
      router.push("/dashboard");
    }, 3000); // Show success animation for 3 seconds
  };

  const handleSkipVerification = () => {
    // Allow user to skip verification and proceed to dashboard
    router.push("/dashboard");
  };

  // Show different flows based on registration state
  if (registrationFlow === 'verification') {
    return (
      <EmailVerificationFlow
        email={formData.email}
        onVerificationComplete={handleVerificationComplete}
        onSkip={handleSkipVerification}
      />
    );
  }

  if (registrationFlow === 'success') {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardContent className="pt-6">
          <SuccessAnimation
            title="Welcome to NovaCron!"
            description="Your account has been created successfully. You will be redirected to your dashboard shortly."
          />
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between mb-4">
          <div>
            <CardTitle>Create Your NovaCron Account</CardTitle>
            <CardDescription>
              Step {currentStep} of {totalSteps}: {getStepInfo()[currentStep - 1]?.description}
            </CardDescription>
          </div>
          <div className="text-2xl font-bold text-blue-600">
            {currentStep}/{totalSteps}
          </div>
        </div>
        <Progress value={progress} className="h-2" />
      </CardHeader>
      
      <CardContent>
        {/* Error display */}
        {errors.length > 0 && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <div className="flex items-start gap-2">
              <svg className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                  Please fix the following errors:
                </h3>
                <ul className="mt-1 text-sm text-red-700 dark:text-red-300 list-disc list-inside">
                  {errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
        
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {/* Step 1: Account Type */}
            {currentStep === 1 && (
              <div className="space-y-4">
                <RadioGroup
                  value={formData.accountType}
                  onValueChange={(value) => updateFormData('accountType', value)}
                >
                  <div className="grid gap-4">
                    <label
                      htmlFor="personal"
                      className={cn(
                        "flex items-start space-x-3 p-4 border-2 rounded-lg cursor-pointer transition-all",
                        formData.accountType === 'personal'
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
                      )}
                    >
                      <RadioGroupItem value="personal" id="personal" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">👤</span>
                          <div>
                            <h3 className="font-semibold">Personal Account</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              For individual developers and small projects
                            </p>
                          </div>
                        </div>
                        <ul className="mt-2 text-sm text-gray-600 dark:text-gray-400 space-y-1">
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Up to 5 VMs
                          </li>
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Basic monitoring
                          </li>
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Community support
                          </li>
                        </ul>
                      </div>
                    </label>
                    
                    <label
                      htmlFor="organization"
                      className={cn(
                        "flex items-start space-x-3 p-4 border-2 rounded-lg cursor-pointer transition-all",
                        formData.accountType === 'organization'
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-gray-200 dark:border-gray-700 hover:border-gray-300"
                      )}
                    >
                      <RadioGroupItem value="organization" id="organization" />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-2xl">🏢</span>
                          <div>
                            <h3 className="font-semibold">Organization Account</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              For teams and enterprise deployments
                            </p>
                          </div>
                        </div>
                        <ul className="mt-2 text-sm text-gray-600 dark:text-gray-400 space-y-1">
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Unlimited VMs
                          </li>
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Advanced analytics
                          </li>
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Priority support
                          </li>
                          <li className="flex items-center gap-1">
                            <span className="text-green-500">✓</span> Team management
                          </li>
                        </ul>
                      </div>
                    </label>
                  </div>
                </RadioGroup>
              </div>
            )}
            
            {/* Step 2: Personal Information */}
            {currentStep === 2 && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="firstName">First Name *</Label>
                    <Input
                      id="firstName"
                      placeholder="John"
                      value={formData.firstName}
                      onChange={(e) => updateFormData('firstName', e.target.value)}
                      className={errors.some(e => e.includes("First name")) ? "border-red-500" : ""}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last Name *</Label>
                    <Input
                      id="lastName"
                      placeholder="Doe"
                      value={formData.lastName}
                      onChange={(e) => updateFormData('lastName', e.target.value)}
                      className={errors.some(e => e.includes("Last name")) ? "border-red-500" : ""}
                    />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="email">Email Address *</Label>
                  <div className="relative">
                    <Input
                      id="email"
                      type="email"
                      placeholder="user@organization.com"
                      value={formData.email}
                      onChange={(e) => updateFormData('email', e.target.value)}
                      className={cn(
                        errors.some(e => e.includes("email")) ? "border-red-500" : "",
                        emailAvailable === false ? "border-red-500" : "",
                        emailAvailable === true ? "border-green-500" : ""
                      )}
                    />
                    {checkingEmail && (
                      <div className="absolute right-3 top-3">
                        <Icons.spinner className="h-4 w-4 animate-spin text-gray-400" />
                      </div>
                    )}
                    {!checkingEmail && emailAvailable === true && (
                      <div className="absolute right-3 top-3">
                        <svg className="h-4 w-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                    {!checkingEmail && emailAvailable === false && (
                      <div className="absolute right-3 top-3">
                        <svg className="h-4 w-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                  </div>
                  {emailAvailable === false && (
                    <p className="text-sm text-red-600">This email is already registered</p>
                  )}
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number (Optional)</Label>
                  <Input
                    id="phone"
                    type="tel"
                    placeholder="+1 (555) 123-4567"
                    value={formData.phone}
                    onChange={(e) => updateFormData('phone', e.target.value)}
                  />
                  <p className="text-xs text-gray-500">For account recovery and security notifications</p>
                </div>
              </div>
            )}
            
            {/* Step 3: Organization Details (conditional) */}
            {currentStep === 3 && formData.accountType === 'organization' && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="orgName">Organization Name *</Label>
                  <Input
                    id="orgName"
                    placeholder="Acme Corporation"
                    value={formData.organizationName}
                    onChange={(e) => updateFormData('organizationName', e.target.value)}
                    className={errors.some(e => e.includes("Organization")) ? "border-red-500" : ""}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="orgSize">Organization Size *</Label>
                  <Select
                    value={formData.organizationSize}
                    onValueChange={(value) => updateFormData('organizationSize', value)}
                  >
                    <SelectTrigger
                      id="orgSize"
                      className={errors.some(e => e.includes("size")) ? "border-red-500" : ""}
                    >
                      <SelectValue placeholder="Select organization size" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1-10">1-10 employees</SelectItem>
                      <SelectItem value="11-50">11-50 employees</SelectItem>
                      <SelectItem value="51-200">51-200 employees</SelectItem>
                      <SelectItem value="201-500">201-500 employees</SelectItem>
                      <SelectItem value="501-1000">501-1000 employees</SelectItem>
                      <SelectItem value="1000+">1000+ employees</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                  <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
                    Organization Benefits
                  </h4>
                  <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                    <li className="flex items-center gap-2">
                      <span className="text-blue-600">✓</span> Centralized billing
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-blue-600">✓</span> Team member management
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-blue-600">✓</span> Advanced access controls
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-blue-600">✓</span> Dedicated support channel
                    </li>
                  </ul>
                </div>
              </div>
            )}
            
            {/* Step 4: Security */}
            {((currentStep === 3 && formData.accountType !== 'organization') || 
              (currentStep === 4 && formData.accountType === 'organization')) && (
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="password">Password *</Label>
                  <PasswordInput
                    id="password"
                    placeholder="Enter a strong password"
                    value={formData.password}
                    onChange={(e) => updateFormData('password', e.target.value)}
                    className={errors.some(e => e.includes("Password")) ? "border-red-500" : ""}
                    autoComplete="new-password"
                  />
                  <PasswordStrengthIndicator password={formData.password} />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm Password *</Label>
                  <PasswordInput
                    id="confirmPassword"
                    placeholder="Re-enter your password"
                    value={formData.confirmPassword}
                    onChange={(e) => updateFormData('confirmPassword', e.target.value)}
                    className={errors.some(e => e.includes("match")) ? "border-red-500" : ""}
                    autoComplete="new-password"
                  />
                  {formData.confirmPassword && formData.password !== formData.confirmPassword && (
                    <p className="text-sm text-red-600">Passwords do not match</p>
                  )}
                </div>
                
                <div className="space-y-3 pt-2">
                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="twoFactor"
                      checked={formData.enableTwoFactor}
                      onCheckedChange={(checked) => updateFormData('enableTwoFactor', checked)}
                    />
                    <div className="space-y-1">
                      <label
                        htmlFor="twoFactor"
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        Enable Two-Factor Authentication
                      </label>
                      <p className="text-xs text-gray-500">
                        Add an extra layer of security to your account (recommended)
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start space-x-3">
                    <Checkbox
                      id="terms"
                      checked={formData.acceptTerms}
                      onCheckedChange={(checked) => updateFormData('acceptTerms', checked as boolean)}
                      className={errors.some(e => e.includes("terms")) ? "border-red-500" : ""}
                    />
                    <div className="space-y-1">
                      <label
                        htmlFor="terms"
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        I accept the Terms of Service and Privacy Policy *
                      </label>
                      <p className="text-xs text-gray-500">
                        By creating an account, you agree to our{" "}
                        <a href="/terms" className="text-blue-600 hover:underline">Terms of Service</a>
                        {" "}and{" "}
                        <a href="/privacy" className="text-blue-600 hover:underline">Privacy Policy</a>
                      </p>
                    </div>
                  </div>
                </div>
                
                {formData.enableTwoFactor && (
                  <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <p className="text-sm text-green-800 dark:text-green-200">
                      📱 You'll set up two-factor authentication after creating your account
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
        
        {/* Navigation buttons */}
        <div className="flex justify-between mt-6 pt-4 border-t">
          <Button
            type="button"
            variant="outline"
            onClick={handleBack}
            disabled={currentStep === 1 || isLoading}
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </Button>
          
          <Button
            type="button"
            onClick={handleNext}
            disabled={isLoading}
            className="min-w-[120px]"
          >
            {isLoading ? (
              <>
                <Icons.spinner className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : currentStep === totalSteps ? (
              <>
                Complete Registration
                <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </>
            ) : (
              <>
                Next
                <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}