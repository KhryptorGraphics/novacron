module github.com/khryptorgraphics/novacron

go 1.23.0

toolchain go1.24.4

require (
	github.com/digitalocean/go-libvirt v0.0.0-20250616175656-5843751af96c // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/gorilla/handlers v1.5.2
	github.com/gorilla/mux v1.8.1
	github.com/gorilla/websocket v1.5.3
)

replace github.com/khryptorgraphics/novacron/backend/core => ./backend/core

require github.com/khryptorgraphics/novacron/backend/core v0.0.0-00010101000000-000000000000

require (
	github.com/felixge/httpsnoop v1.0.3 // indirect
	github.com/klauspost/compress v1.17.7 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/crypto v0.39.0 // indirect
	golang.org/x/sys v0.33.0 // indirect
)
