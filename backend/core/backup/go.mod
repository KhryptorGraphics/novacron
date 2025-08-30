module github.com/khryptorgraphics/novacron/backend/core/backup

go 1.23.0

toolchain go1.24.6

require github.com/khryptorgraphics/novacron/backend/core v0.0.0

require (
	github.com/golang-jwt/jwt/v5 v5.3.0 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/klauspost/compress v1.17.7 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/crypto v0.36.0 // indirect
	golang.org/x/sys v0.31.0 // indirect
)

replace github.com/khryptorgraphics/novacron/backend/core => ../
