module github.com/khryptorgraphics/novacron/backend/core/orchestration

go 1.23.0

require (
	github.com/google/uuid v1.6.0
	github.com/nats-io/nats.go v1.37.0
	github.com/sirupsen/logrus v1.9.3
	github.com/stretchr/testify v1.11.1
	gopkg.in/yaml.v2 v2.4.0
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/klauspost/compress v1.18.0 // indirect
	github.com/nats-io/nkeys v0.4.9 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/crypto v0.41.0 // indirect
	golang.org/x/sys v0.35.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/khryptorgraphics/novacron/backend/core => ../
