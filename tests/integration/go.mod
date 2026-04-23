module github.com/khryptorgraphics/novacron/tests/integration

go 1.24.0

toolchain go1.24.6

replace github.com/khryptorgraphics/novacron => ../..
replace github.com/khryptorgraphics/novacron/backend/core => ../../backend/core
replace github.com/novacron-org/novacron/backend/core => ../../backend/core
replace github.com/novacron/backend/core => ../../backend/core

require (
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/khryptorgraphics/novacron v0.0.0-00010101000000-000000000000
	github.com/lib/pq v1.10.9
	github.com/stretchr/testify v1.9.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
