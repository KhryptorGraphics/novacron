package auth

import (
	"fmt"
	"time"
)

const authTestDefaultUsername = "test_user"

func authTestEmail() string {
	return "test@example.com"
}

func authGeneratedEmail() string {
	return fmt.Sprintf("test-%d@example.com", time.Now().UnixNano())
}
