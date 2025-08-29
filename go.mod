module github.com/khryptorgraphics/novacron

go 1.24.0

toolchain go1.24.6

require (
	github.com/digitalocean/go-libvirt v0.0.0-20250616175656-5843751af96c // indirect
	github.com/google/uuid v1.6.0
	github.com/gorilla/handlers v1.5.2
	github.com/gorilla/mux v1.8.1
	github.com/gorilla/websocket v1.5.3
)

require (
	cloud.google.com/go/compute v1.44.0
	github.com/Azure/azure-sdk-for-go v68.0.0+incompatible
	github.com/Azure/go-autorest/autorest v0.11.30
	github.com/Azure/go-autorest/autorest/azure/auth v0.5.13
	github.com/aws/aws-sdk-go v1.55.8
	github.com/go-redis/redis/v8 v8.11.5
	github.com/golang-jwt/jwt/v5 v5.3.0
	github.com/jmoiron/sqlx v1.4.0
	github.com/khryptorgraphics/novacron/backend/core v0.0.0-00010101000000-000000000000
	github.com/khryptorgraphics/novacron/backend/pkg/logger v0.0.0-00010101000000-000000000000
	github.com/lib/pq v1.10.9
	github.com/sirupsen/logrus v1.9.3
	github.com/sony/gobreaker v1.0.0
	github.com/stretchr/testify v1.10.0
	google.golang.org/api v0.248.0
	google.golang.org/genproto v0.0.0-20250826171959-ef028d996bc1
	google.golang.org/protobuf v1.36.8
)

require (
	cloud.google.com/go/auth v0.16.5 // indirect
	cloud.google.com/go/auth/oauth2adapt v0.2.8 // indirect
	cloud.google.com/go/compute/metadata v0.8.0 // indirect
	github.com/Azure/go-autorest v14.2.0+incompatible // indirect
	github.com/Azure/go-autorest/autorest/adal v0.9.22 // indirect
	github.com/Azure/go-autorest/autorest/azure/cli v0.4.6 // indirect
	github.com/Azure/go-autorest/autorest/date v0.3.0 // indirect
	github.com/Azure/go-autorest/autorest/to v0.4.1 // indirect
	github.com/Azure/go-autorest/autorest/validation v0.3.2 // indirect
	github.com/Azure/go-autorest/logger v0.2.1 // indirect
	github.com/Azure/go-autorest/tracing v0.6.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/dimchansky/utfbom v1.1.1 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/golang-jwt/jwt/v4 v4.5.2 // indirect
	github.com/google/s2a-go v0.1.9 // indirect
	github.com/googleapis/enterprise-certificate-proxy v0.3.6 // indirect
	github.com/googleapis/gax-go/v2 v2.15.0 // indirect
	github.com/jmespath/go-jmespath v0.4.0 // indirect
	github.com/klauspost/compress v1.17.7 // indirect
	github.com/mitchellh/go-homedir v1.1.0 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.61.0 // indirect
	go.opentelemetry.io/otel v1.36.0 // indirect
	go.opentelemetry.io/otel/metric v1.36.0 // indirect
	go.opentelemetry.io/otel/trace v1.36.0 // indirect
	golang.org/x/crypto v0.41.0 // indirect
	golang.org/x/net v0.43.0 // indirect
	golang.org/x/oauth2 v0.30.0 // indirect
	golang.org/x/sys v0.35.0 // indirect
	golang.org/x/text v0.28.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20250818200422-3122310a409c // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250818200422-3122310a409c // indirect
	google.golang.org/grpc v1.74.2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/khryptorgraphics/novacron/backend/core => ./backend/core

replace github.com/khryptorgraphics/novacron/backend/pkg/logger => ./backend/pkg/logger
