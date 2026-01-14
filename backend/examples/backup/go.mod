module github.com/khryptorgraphics/novacron/backend/examples/backup

go 1.21

require (
	github.com/khryptorgraphics/novacron v0.0.0
	github.com/khryptorgraphics/novacron/backend/core/backup v0.0.0
	github.com/khryptorgraphics/novacron/backend/core/backup/providers v0.0.0
)

replace (
	github.com/khryptorgraphics/novacron => ../../../
	github.com/khryptorgraphics/novacron/backend/core/backup => ../../core/backup
	github.com/khryptorgraphics/novacron/backend/core/backup/providers => ../../core/backup/providers
)
