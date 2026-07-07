import {
  validateRegistrationStep,
  validatePassword,
  validateEmail,
  type RegistrationData,
} from '@/lib/validation';

describe('Validation Functions', () => {
  describe('validateEmail', () => {
    it('accepts correct email addresses', () => {
      const validEmails = [
        'user@example.com',
        'test.email+tag@domain.co.uk',
        'firstname-lastname@example.com',
        'user123@test-domain.com',
      ];

      validEmails.forEach((email) => {
        expect(validateEmail(email).isValid).toBe(true);
      });
    });

    it('rejects clearly invalid email addresses', () => {
      const invalidEmails = ['invalid-email', '@example.com', 'user@', ''];

      invalidEmails.forEach((email) => {
        expect(validateEmail(email).isValid).toBe(false);
      });
    });
  });

  describe('validatePassword', () => {
    it('scores strong passwords at the top of the range', () => {
      const strongPasswords = ['Xk9$mL2#nQ7w', 'Zt5&pR8!wY3q', 'Bv6@kN4^dM9r', 'Hf2#sW7$gJ5x'];

      strongPasswords.forEach((password) => {
        expect(validatePassword(password).score).toBeGreaterThanOrEqual(4);
      });
    });

    it('scores weak passwords low', () => {
      const weakPasswords = ['password', '123456', 'abc123', 'Password'];

      weakPasswords.forEach((password) => {
        expect(validatePassword(password).score).toBeLessThan(4);
      });
    });

    it('returns actionable suggestions for a poor password', () => {
      const suggestions = validatePassword('short').suggestions.join(' ');

      expect(suggestions).toMatch(/12 characters/i);
      expect(suggestions).toMatch(/uppercase/i);
      expect(suggestions).toMatch(/number/i);
      expect(suggestions).toMatch(/special/i);
    });

    it('handles an empty password', () => {
      const result = validatePassword('');

      expect(result.score).toBe(0);
      expect(result.feedback).toContain('required');
    });
  });

  describe('validateRegistrationStep', () => {
    const validPersonalData: RegistrationData = {
      accountType: 'personal',
      firstName: 'John',
      lastName: 'Doe',
      email: 'john.doe@example.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      acceptTerms: true,
    };

    const validOrgData: RegistrationData = {
      accountType: 'organization',
      firstName: 'Jane',
      lastName: 'Smith',
      email: 'jane@company.com',
      password: 'SecurePassword123!',
      confirmPassword: 'SecurePassword123!',
      organizationName: 'Test Company',
      organizationSize: '10-50',
      acceptTerms: true,
    };

    it('accepts valid personal information (step 2)', () => {
      const result = validateRegistrationStep(2, validPersonalData);

      expect(result.isValid).toBe(true);
      expect(result.errors).toEqual([]);
    });

    it('accepts a valid security step (step 4)', () => {
      expect(validateRegistrationStep(4, validPersonalData).isValid).toBe(true);
    });

    it('accepts valid organization details (step 3)', () => {
      expect(validateRegistrationStep(3, validOrgData).isValid).toBe(true);
    });

    it('requires a first name (step 2)', () => {
      const result = validateRegistrationStep(2, { ...validPersonalData, firstName: '' });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/first name/i);
    });

    it('requires a valid email (step 2)', () => {
      const result = validateRegistrationStep(2, { ...validPersonalData, email: 'invalid-email' });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/valid email/i);
    });

    it('requires password confirmation to match (step 4)', () => {
      const result = validateRegistrationStep(4, {
        ...validPersonalData,
        confirmPassword: 'DifferentPassword!',
      });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/do not match/i);
    });

    it('requires terms acceptance (step 4)', () => {
      const result = validateRegistrationStep(4, { ...validPersonalData, acceptTerms: false });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/terms/i);
    });

    it('requires an organization name for organization accounts (step 3)', () => {
      const result = validateRegistrationStep(3, { ...validOrgData, organizationName: '' });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/organization name/i);
    });

    it('enforces name length limits (step 2)', () => {
      const result = validateRegistrationStep(2, {
        ...validPersonalData,
        firstName: 'a'.repeat(51),
      });

      expect(result.isValid).toBe(false);
      expect(result.errors.join(' ')).toMatch(/less than 50 characters/i);
    });
  });
});
