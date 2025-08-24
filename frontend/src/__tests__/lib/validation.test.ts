import { validateRegistrationData, validatePassword, validateEmail } from '@/lib/validation';

describe('Validation Functions', () => {
  describe('validateEmail', () => {
    it('validates correct email addresses', () => {
      const validEmails = [
        'user@example.com',
        'test.email+tag@domain.co.uk',
        'firstname-lastname@example.com',
        'user123@test-domain.com',
      ];

      validEmails.forEach(email => {
        expect(validateEmail(email)).toBe(true);
      });
    });

    it('rejects invalid email addresses', () => {
      const invalidEmails = [
        'invalid-email',
        '@example.com',
        'user@',
        'user..double.dot@example.com',
        'user@.example.com',
        '',
      ];

      invalidEmails.forEach(email => {
        expect(validateEmail(email)).toBe(false);
      });
    });
  });

  describe('validatePassword', () => {
    it('validates strong passwords', () => {
      const strongPasswords = [
        'Password123!',
        'ComplexP@ssw0rd',
        'MyStr0ng#Pass',
        'Secure1234$',
      ];

      strongPasswords.forEach(password => {
        const result = validatePassword(password);
        expect(result.isValid).toBe(true);
        expect(result.score).toBeGreaterThanOrEqual(4);
      });
    });

    it('identifies weak passwords', () => {
      const weakPasswords = [
        'password',
        '123456',
        'abc123',
        'Password',
      ];

      weakPasswords.forEach(password => {
        const result = validatePassword(password);
        expect(result.isValid).toBe(false);
        expect(result.score).toBeLessThan(4);
      });
    });

    it('provides specific feedback', () => {
      const result = validatePassword('short');

      expect(result.feedback).toContain('at least 8 characters');
      expect(result.feedback).toContain('uppercase letter');
      expect(result.feedback).toContain('number');
      expect(result.feedback).toContain('special character');
    });

    it('handles empty password', () => {
      const result = validatePassword('');

      expect(result.isValid).toBe(false);
      expect(result.score).toBe(0);
      expect(result.feedback).toContain('required');
    });
  });

  describe('validateRegistrationData', () => {
    const validPersonalData = {
      accountType: 'personal' as const,
      firstName: 'John',
      lastName: 'Doe',
      email: 'john.doe@example.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      acceptTerms: true,
    };

    const validOrgData = {
      accountType: 'organization' as const,
      firstName: 'Jane',
      lastName: 'Smith',
      email: 'jane@company.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      organizationName: 'Test Company',
      organizationSize: '10-50',
      acceptTerms: true,
    };

    it('validates correct personal account data', () => {
      const result = validateRegistrationData(validPersonalData);

      expect(result.isValid).toBe(true);
      expect(result.errors).toEqual({});
    });

    it('validates correct organization account data', () => {
      const result = validateRegistrationData(validOrgData);

      expect(result.isValid).toBe(true);
      expect(result.errors).toEqual({});
    });

    it('requires first name', () => {
      const data = { ...validPersonalData, firstName: '' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.firstName).toContain('required');
    });

    it('requires valid email', () => {
      const data = { ...validPersonalData, email: 'invalid-email' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.email).toContain('valid email');
    });

    it('requires password confirmation to match', () => {
      const data = { ...validPersonalData, confirmPassword: 'DifferentPassword!' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.confirmPassword).toContain('match');
    });

    it('requires terms acceptance', () => {
      const data = { ...validPersonalData, acceptTerms: false };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.acceptTerms).toContain('accept');
    });

    it('requires organization name for organization accounts', () => {
      const data = { ...validOrgData, organizationName: '' };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.organizationName).toContain('required');
    });

    it('validates name length limits', () => {
      const longName = 'a'.repeat(51);
      const data = { ...validPersonalData, firstName: longName };
      const result = validateRegistrationData(data);

      expect(result.isValid).toBe(false);
      expect(result.errors.firstName).toContain('50 characters');
    });
  });
});
