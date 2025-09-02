module github.com/khryptorgraphics/novacron/backend/api/backup

go 1.23.0

toolchain go1.24.6

require (
	github.com/gorilla/mux v1.8.1
	github.com/khryptorgraphics/novacron/backend/core/backup v0.0.0-20250830173050-fe55263834f3
)

replace github.com/khryptorgraphics/novacron/backend/core => ../../core
