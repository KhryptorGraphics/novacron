package auth

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"html/template"
	"net/smtp"
	"strings"
	"time"
)

// EmailConfig defines email service configuration
type EmailConfig struct {
	// SMTPHost is the SMTP server hostname
	SMTPHost string
	// SMTPPort is the SMTP server port (typically 587 for TLS, 465 for SSL, 25 for unencrypted)
	SMTPPort int
	// Username for SMTP authentication
	Username string
	// Password for SMTP authentication
	Password string
	// FromAddress is the sender email address
	FromAddress string
	// FromName is the sender display name
	FromName string
	// UseTLS enables STARTTLS
	UseTLS bool
	// UseSSL enables implicit SSL (port 465)
	UseSSL bool
	// SkipTLSVerify skips TLS certificate verification (not recommended for production)
	SkipTLSVerify bool
	// FrontendURL is the base URL for links in emails
	FrontendURL string
}

// DefaultEmailConfig returns a default email configuration
func DefaultEmailConfig() EmailConfig {
	return EmailConfig{
		SMTPHost:      "localhost",
		SMTPPort:      587,
		FromAddress:   "notifications@giggahost.com",
		FromName:      "NovaCron",
		UseTLS:        true,
		FrontendURL:   "http://localhost:8092",
	}
}

// EmailService handles sending emails
type EmailService struct {
	config    EmailConfig
	templates map[string]*template.Template
}

// NewEmailService creates a new email service
func NewEmailService(config EmailConfig) *EmailService {
	service := &EmailService{
		config:    config,
		templates: make(map[string]*template.Template),
	}

	// Pre-load templates
	service.loadTemplates()

	return service
}

// loadTemplates loads email templates
func (s *EmailService) loadTemplates() {
	// Password Reset Template
	s.templates["password_reset"] = template.Must(template.New("password_reset").Parse(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Password Reset</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .button { display: inline-block; background: #0066cc; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Password Reset Request</h2>
        <p>Hello {{.Username}},</p>
        <p>We received a request to reset your password for your NovaCron account. Click the button below to reset your password:</p>
        <p><a href="{{.ResetURL}}" class="button">Reset Password</a></p>
        <p>If you didn't request this, you can safely ignore this email. The link will expire in {{.ExpiresIn}} minutes.</p>
        <p>For security, this link can only be used once.</p>
        <div class="footer">
            <p>This is an automated message from NovaCron. Please do not reply to this email.</p>
            <p>&copy; {{.Year}} NovaCron. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
`))

	// Account Verification Template
	s.templates["account_verification"] = template.Must(template.New("account_verification").Parse(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Verify Your Account</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .button { display: inline-block; background: #28a745; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Welcome to NovaCron!</h2>
        <p>Hello {{.Username}},</p>
        <p>Thank you for creating a NovaCron account. Please verify your email address by clicking the button below:</p>
        <p><a href="{{.VerificationURL}}" class="button">Verify Email</a></p>
        <p>This link will expire in {{.ExpiresIn}} hours.</p>
        <div class="footer">
            <p>This is an automated message from NovaCron. Please do not reply to this email.</p>
            <p>&copy; {{.Year}} NovaCron. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
`))

	// 2FA Backup Codes Template
	s.templates["2fa_backup_codes"] = template.Must(template.New("2fa_backup_codes").Parse(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Your 2FA Backup Codes</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .codes { background: #f5f5f5; padding: 15px; border-radius: 4px; font-family: monospace; font-size: 16px; }
        .code { display: inline-block; padding: 5px 10px; margin: 5px; background: white; border: 1px solid #ddd; border-radius: 3px; }
        .warning { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin: 15px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Your 2FA Backup Codes</h2>
        <p>Hello {{.Username}},</p>
        <p>You have enabled two-factor authentication on your NovaCron account. Below are your backup codes:</p>
        <div class="codes">
            {{range .BackupCodes}}
            <span class="code">{{.}}</span>
            {{end}}
        </div>
        <div class="warning">
            <strong>Important:</strong> Store these codes in a safe place. Each code can only be used once. If you lose access to your authenticator app, you can use these codes to sign in.
        </div>
        <div class="footer">
            <p>This is an automated message from NovaCron. Please do not reply to this email.</p>
            <p>&copy; {{.Year}} NovaCron. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
`))

	// Login Notification Template
	s.templates["login_notification"] = template.Must(template.New("login_notification").Parse(`
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>New Login Detected</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .details { background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 15px 0; }
        .warning { background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin: 15px 0; }
        .button { display: inline-block; background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>New Login to Your Account</h2>
        <p>Hello {{.Username}},</p>
        <p>We detected a new login to your NovaCron account:</p>
        <div class="details">
            <p><strong>Time:</strong> {{.LoginTime}}</p>
            <p><strong>IP Address:</strong> {{.IPAddress}}</p>
            <p><strong>Location:</strong> {{.Location}}</p>
            <p><strong>Device:</strong> {{.UserAgent}}</p>
        </div>
        <p>If this was you, you can ignore this email.</p>
        <div class="warning">
            <p><strong>Not you?</strong> If you did not log in, your account may be compromised.</p>
            <p><a href="{{.SecurityURL}}" class="button">Secure Your Account</a></p>
        </div>
        <div class="footer">
            <p>This is an automated message from NovaCron. Please do not reply to this email.</p>
            <p>&copy; {{.Year}} NovaCron. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
`))
}

// Send sends an email using the configured SMTP settings
func (s *EmailService) Send(to, subject, body string) error {
	return s.SendWithHTML(to, subject, body, "")
}

// SendWithHTML sends an email with both plain text and HTML body
func (s *EmailService) SendWithHTML(to, subject, textBody, htmlBody string) error {
	from := s.config.FromAddress
	if s.config.FromName != "" {
		from = fmt.Sprintf("%s <%s>", s.config.FromName, s.config.FromAddress)
	}

	// Build the message
	var msg bytes.Buffer
	msg.WriteString(fmt.Sprintf("From: %s\r\n", from))
	msg.WriteString(fmt.Sprintf("To: %s\r\n", to))
	msg.WriteString(fmt.Sprintf("Subject: %s\r\n", subject))
	msg.WriteString("MIME-Version: 1.0\r\n")

	if htmlBody != "" {
		boundary := "=_NovaCron_Email_Boundary_="
		msg.WriteString(fmt.Sprintf("Content-Type: multipart/alternative; boundary=\"%s\"\r\n", boundary))
		msg.WriteString("\r\n")

		// Plain text part
		msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
		msg.WriteString("Content-Type: text/plain; charset=UTF-8\r\n")
		msg.WriteString("Content-Transfer-Encoding: 7bit\r\n\r\n")
		msg.WriteString(textBody)
		msg.WriteString("\r\n")

		// HTML part
		msg.WriteString(fmt.Sprintf("--%s\r\n", boundary))
		msg.WriteString("Content-Type: text/html; charset=UTF-8\r\n")
		msg.WriteString("Content-Transfer-Encoding: 7bit\r\n\r\n")
		msg.WriteString(htmlBody)
		msg.WriteString("\r\n")

		msg.WriteString(fmt.Sprintf("--%s--\r\n", boundary))
	} else {
		msg.WriteString("Content-Type: text/plain; charset=UTF-8\r\n")
		msg.WriteString("\r\n")
		msg.WriteString(textBody)
	}

	// Connect to SMTP server
	addr := fmt.Sprintf("%s:%d", s.config.SMTPHost, s.config.SMTPPort)

	auth := smtp.PlainAuth("", s.config.Username, s.config.Password, s.config.SMTPHost)

	if s.config.UseSSL {
		// Implicit TLS (port 465)
		return s.sendWithSSL(addr, auth, s.config.FromAddress, to, msg.Bytes())
	}

	if s.config.UseTLS {
		// STARTTLS (port 587)
		return s.sendWithTLS(addr, auth, s.config.FromAddress, to, msg.Bytes())
	}

	// Plain SMTP
	return smtp.SendMail(addr, auth, s.config.FromAddress, []string{to}, msg.Bytes())
}

// sendWithTLS sends email using STARTTLS
func (s *EmailService) sendWithTLS(addr string, auth smtp.Auth, from, to string, msg []byte) error {
	client, err := smtp.Dial(addr)
	if err != nil {
		return fmt.Errorf("failed to connect to SMTP server: %w", err)
	}
	defer client.Close()

	// Send STARTTLS
	tlsConfig := &tls.Config{
		ServerName:         s.config.SMTPHost,
		InsecureSkipVerify: s.config.SkipTLSVerify,
	}
	if err := client.StartTLS(tlsConfig); err != nil {
		return fmt.Errorf("failed to start TLS: %w", err)
	}

	// Authenticate
	if auth != nil {
		if err := client.Auth(auth); err != nil {
			return fmt.Errorf("failed to authenticate: %w", err)
		}
	}

	// Send email
	if err := client.Mail(from); err != nil {
		return fmt.Errorf("failed to set sender: %w", err)
	}
	if err := client.Rcpt(to); err != nil {
		return fmt.Errorf("failed to set recipient: %w", err)
	}

	w, err := client.Data()
	if err != nil {
		return fmt.Errorf("failed to start data: %w", err)
	}

	_, err = w.Write(msg)
	if err != nil {
		return fmt.Errorf("failed to write message: %w", err)
	}

	if err := w.Close(); err != nil {
		return fmt.Errorf("failed to close data writer: %w", err)
	}

	return client.Quit()
}

// sendWithSSL sends email using implicit SSL
func (s *EmailService) sendWithSSL(addr string, auth smtp.Auth, from, to string, msg []byte) error {
	tlsConfig := &tls.Config{
		ServerName:         s.config.SMTPHost,
		InsecureSkipVerify: s.config.SkipTLSVerify,
	}

	conn, err := tls.Dial("tcp", addr, tlsConfig)
	if err != nil {
		return fmt.Errorf("failed to connect with SSL: %w", err)
	}

	client, err := smtp.NewClient(conn, s.config.SMTPHost)
	if err != nil {
		return fmt.Errorf("failed to create SMTP client: %w", err)
	}
	defer client.Close()

	// Authenticate
	if auth != nil {
		if err := client.Auth(auth); err != nil {
			return fmt.Errorf("failed to authenticate: %w", err)
		}
	}

	// Send email
	if err := client.Mail(from); err != nil {
		return fmt.Errorf("failed to set sender: %w", err)
	}
	if err := client.Rcpt(to); err != nil {
		return fmt.Errorf("failed to set recipient: %w", err)
	}

	w, err := client.Data()
	if err != nil {
		return fmt.Errorf("failed to start data: %w", err)
	}

	_, err = w.Write(msg)
	if err != nil {
		return fmt.Errorf("failed to write message: %w", err)
	}

	if err := w.Close(); err != nil {
		return fmt.Errorf("failed to close data writer: %w", err)
	}

	return client.Quit()
}

// SendPasswordReset sends a password reset email
func (s *EmailService) SendPasswordReset(to, username, resetToken string, expiresInMinutes int) error {
	tmpl, ok := s.templates["password_reset"]
	if !ok {
		return fmt.Errorf("password reset template not found")
	}

	data := map[string]interface{}{
		"Username":  username,
		"ResetURL":  fmt.Sprintf("%s/auth/reset-password?token=%s", s.config.FrontendURL, resetToken),
		"ExpiresIn": expiresInMinutes,
		"Year":      time.Now().Year(),
	}

	var htmlBuf bytes.Buffer
	if err := tmpl.Execute(&htmlBuf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	textBody := fmt.Sprintf(
		"Hello %s,\n\n"+
			"We received a request to reset your password for your NovaCron account.\n\n"+
			"Reset your password: %s/auth/reset-password?token=%s\n\n"+
			"This link will expire in %d minutes.\n\n"+
			"If you didn't request this, you can safely ignore this email.\n",
		username, s.config.FrontendURL, resetToken, expiresInMinutes,
	)

	return s.SendWithHTML(to, "Reset Your NovaCron Password", textBody, htmlBuf.String())
}

// SendAccountVerification sends an account verification email
func (s *EmailService) SendAccountVerification(to, username, verificationToken string, expiresInHours int) error {
	tmpl, ok := s.templates["account_verification"]
	if !ok {
		return fmt.Errorf("account verification template not found")
	}

	data := map[string]interface{}{
		"Username":        username,
		"VerificationURL": fmt.Sprintf("%s/auth/verify-email?token=%s", s.config.FrontendURL, verificationToken),
		"ExpiresIn":       expiresInHours,
		"Year":            time.Now().Year(),
	}

	var htmlBuf bytes.Buffer
	if err := tmpl.Execute(&htmlBuf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	textBody := fmt.Sprintf(
		"Hello %s,\n\n"+
			"Welcome to NovaCron! Please verify your email address.\n\n"+
			"Verify your email: %s/auth/verify-email?token=%s\n\n"+
			"This link will expire in %d hours.\n",
		username, s.config.FrontendURL, verificationToken, expiresInHours,
	)

	return s.SendWithHTML(to, "Verify Your NovaCron Account", textBody, htmlBuf.String())
}

// SendBackupCodes sends 2FA backup codes email
func (s *EmailService) SendBackupCodes(to, username string, backupCodes []string) error {
	tmpl, ok := s.templates["2fa_backup_codes"]
	if !ok {
		return fmt.Errorf("2FA backup codes template not found")
	}

	data := map[string]interface{}{
		"Username":    username,
		"BackupCodes": backupCodes,
		"Year":        time.Now().Year(),
	}

	var htmlBuf bytes.Buffer
	if err := tmpl.Execute(&htmlBuf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	textBody := fmt.Sprintf(
		"Hello %s,\n\n"+
			"You have enabled two-factor authentication on your NovaCron account.\n\n"+
			"Your backup codes:\n%s\n\n"+
			"Store these codes in a safe place. Each code can only be used once.\n",
		username, strings.Join(backupCodes, "\n"),
	)

	return s.SendWithHTML(to, "Your NovaCron 2FA Backup Codes", textBody, htmlBuf.String())
}

// SendLoginNotification sends a login notification email
func (s *EmailService) SendLoginNotification(to, username, ipAddress, location, userAgent string, loginTime time.Time) error {
	tmpl, ok := s.templates["login_notification"]
	if !ok {
		return fmt.Errorf("login notification template not found")
	}

	data := map[string]interface{}{
		"Username":    username,
		"LoginTime":   loginTime.Format("January 2, 2006 at 3:04 PM MST"),
		"IPAddress":   ipAddress,
		"Location":    location,
		"UserAgent":   userAgent,
		"SecurityURL": fmt.Sprintf("%s/settings/security", s.config.FrontendURL),
		"Year":        time.Now().Year(),
	}

	var htmlBuf bytes.Buffer
	if err := tmpl.Execute(&htmlBuf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	textBody := fmt.Sprintf(
		"Hello %s,\n\n"+
			"We detected a new login to your NovaCron account:\n\n"+
			"Time: %s\n"+
			"IP Address: %s\n"+
			"Location: %s\n"+
			"Device: %s\n\n"+
			"If this was you, you can ignore this email.\n\n"+
			"If this wasn't you, secure your account: %s/settings/security\n",
		username, loginTime.Format("January 2, 2006 at 3:04 PM MST"),
		ipAddress, location, userAgent, s.config.FrontendURL,
	)

	return s.SendWithHTML(to, "New Login to Your NovaCron Account", textBody, htmlBuf.String())
}
