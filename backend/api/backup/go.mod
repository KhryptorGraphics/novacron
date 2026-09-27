module github.com/khryptorgraphics/novacron/backend/api/backup

go 1.24.0

toolchain go1.24.6

require (
	github.com/gorilla/mux v1.8.1
	github.com/khryptorgraphics/novacron/backend/core/backup v0.0.0
)

require (
	github.com/chmduquesne/rollinghash v4.0.0+incompatible // indirect
	github.com/khryptorgraphics/novacron/backend/core v0.0.0 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
)

replace (
	github.com/khryptorgraphics/novacron/backend/core => ../../core
	github.com/khryptorgraphics/novacron/backend/core/backup => ../../core/backup
)
