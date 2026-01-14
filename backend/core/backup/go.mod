module github.com/khryptorgraphics/novacron/backend/core/backup

go 1.24.0

toolchain go1.24.6

require (
	github.com/chmduquesne/rollinghash v4.0.0+incompatible
	github.com/khryptorgraphics/novacron/backend/core v0.0.0
	github.com/klauspost/compress v1.18.1
)

replace github.com/khryptorgraphics/novacron/backend/core => ../
