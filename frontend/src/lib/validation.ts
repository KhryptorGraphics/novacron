// Form validation utilities for NovaCron

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

export interface PasswordStrength {
  score: number; // 0-4
  feedback: string;
  suggestions: string[];
}

// Email validation
export const validateEmail = (email: string): ValidationResult => {
  const errors: string[] = [];
  
  if (!email) {
    errors.push("Email is required");
  } else {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      errors.push("Please enter a valid email address");
    }
    
    // Check for common typos in popular domains
    const domain = email.split('@')[1]?.toLowerCase();
    const commonDomains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com'];
    const typos: Record<string, string> = {
      'gmial.com': 'gmail.com',
      'gmai.com': 'gmail.com',
      'yahooo.com': 'yahoo.com',
      'outlok.com': 'outlook.com',
    };
    
    if (domain && typos[domain]) {
      errors.push(`Did you mean ${email.split('@')[0]}@${typos[domain]}?`);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Password strength validation
export const validatePassword = (password: string): PasswordStrength => {
  let score = 0;
  const suggestions: string[] = [];
  
  if (!password) {
    return {
      score: 0,
      feedback: "Password is required",
      suggestions: ["Enter a password"]
    };
  }
  
  // Length check
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  else suggestions.push("Use at least 12 characters for better security");
  
  // Complexity checks
  if (/[a-z]/.test(password)) score += 0.5;
  else suggestions.push("Include lowercase letters");
  
  if (/[A-Z]/.test(password)) score += 0.5;
  else suggestions.push("Include uppercase letters");
  
  if (/[0-9]/.test(password)) score++;
  else suggestions.push("Include numbers");
  
  if (/[^a-zA-Z0-9]/.test(password)) score++;
  else suggestions.push("Include special characters (!@#$%^&*)");
  
  // Common patterns and dictionary check
  const commonPatterns = [
    /^123456/,
    /^password/i,
    /^qwerty/i,
    /^abc123/i,
    /^admin/i,
    /^letmein/i,
    /^welcome/i,
    /^dragon/i,
    /^monkey/i,
    /^sunshine/i
  ];
  
  const hasCommonPattern = commonPatterns.some(pattern => pattern.test(password));
  if (hasCommonPattern) {
    score = Math.max(0, score - 2);
    suggestions.push("Avoid common passwords and dictionary words");
  }

  // Sequential or repeated characters penalty
  const hasSequential = /123|abc|987|zyx/i.test(password);
  const hasRepeated = /(.)\1{2,}/.test(password);
  
  if (hasSequential) {
    score = Math.max(0, score - 1);
    suggestions.push("Avoid sequential characters (123, abc)");
  }
  
  if (hasRepeated) {
    score = Math.max(0, score - 1);
    suggestions.push("Avoid repeating characters (aaa, 111)");
  }
  
  // Determine feedback based on score
  let feedback = "";
  if (score < 1) feedback = "Very Weak";
  else if (score < 2) feedback = "Weak";
  else if (score < 3) feedback = "Fair";
  else if (score < 4) feedback = "Good";
  else feedback = "Strong";
  
  return {
    score: Math.min(4, Math.floor(score)),
    feedback,
    suggestions
  };
};

// Name validation
export const validateName = (name: string, fieldName: string): ValidationResult => {
  const errors: string[] = [];
  
  if (!name) {
    errors.push(`${fieldName} is required`);
  } else {
    if (name.length < 2) {
      errors.push(`${fieldName} must be at least 2 characters`);
    }
    if (name.length > 50) {
      errors.push(`${fieldName} must be less than 50 characters`);
    }
    if (!/^[a-zA-Z\s'-]+$/.test(name)) {
      errors.push(`${fieldName} can only contain letters, spaces, hyphens, and apostrophes`);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Organization validation
export const validateOrganization = (orgName: string): ValidationResult => {
  const errors: string[] = [];
  
  if (!orgName) {
    errors.push("Organization name is required");
  } else {
    if (orgName.length < 2) {
      errors.push("Organization name must be at least 2 characters");
    }
    if (orgName.length > 100) {
      errors.push("Organization name must be less than 100 characters");
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Phone validation (optional)
export const validatePhone = (phone: string): ValidationResult => {
  const errors: string[] = [];
  
  if (phone) {
    // Remove all non-digits for validation
    const digitsOnly = phone.replace(/\D/g, '');
    
    if (digitsOnly.length < 10) {
      errors.push("Phone number must be at least 10 digits");
    }
    if (digitsOnly.length > 15) {
      errors.push("Phone number is too long");
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Terms acceptance validation
export const validateTerms = (accepted: boolean): ValidationResult => {
  const errors: string[] = [];
  
  if (!accepted) {
    errors.push("You must accept the terms and conditions");
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Comprehensive registration form validation
export interface RegistrationData {
  accountType: 'personal' | 'organization' | '';
  firstName: string;
  lastName: string;
  email: string;
  password: string;
  confirmPassword: string;
  organizationName?: string;
  organizationSize?: string;
  phone?: string;
  acceptTerms: boolean;
  enableTwoFactor?: boolean;
}

export const validateRegistrationStep = (
  step: number,
  data: RegistrationData
): ValidationResult => {
  const errors: string[] = [];
  
  switch (step) {
    case 1: // Account Type
      if (!data.accountType) {
        errors.push("Please select an account type");
      }
      break;
      
    case 2: // Personal Information
      const firstNameValidation = validateName(data.firstName, "First name");
      const lastNameValidation = validateName(data.lastName, "Last name");
      const emailValidation = validateEmail(data.email);
      
      errors.push(...firstNameValidation.errors);
      errors.push(...lastNameValidation.errors);
      errors.push(...emailValidation.errors);
      
      if (data.phone) {
        const phoneValidation = validatePhone(data.phone);
        errors.push(...phoneValidation.errors);
      }
      break;
      
    case 3: // Organization Details (if applicable)
      if (data.accountType === 'organization') {
        const orgValidation = validateOrganization(data.organizationName || '');
        errors.push(...orgValidation.errors);
        
        if (!data.organizationSize) {
          errors.push("Please select organization size");
        }
      }
      break;
      
    case 4: // Security
      const passwordStrength = validatePassword(data.password);
      if (passwordStrength.score < 2) {
        errors.push("Password is too weak");
        errors.push(...passwordStrength.suggestions);
      }
      
      if (data.password !== data.confirmPassword) {
        errors.push("Passwords do not match");
      }
      
      const termsValidation = validateTerms(data.acceptTerms);
      errors.push(...termsValidation.errors);
      break;
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

// Debounce utility for real-time validation
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}