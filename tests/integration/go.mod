module github.com/khryptorgraphics/novacron/tests/integration

go 1.25.0

replace github.com/khryptorgraphics/novacron => ../..

replace github.com/khryptorgraphics/novacron/backend/core => ../../backend/core

replace github.com/khryptorgraphics/novacron/backend/core/backup => ../../backend/core/backup

replace github.com/khryptorgraphics/novacron/backend/core/orchestration => ../../backend/core/orchestration

require (
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674
	github.com/khryptorgraphics/novacron/backend/core v0.0.0
	github.com/khryptorgraphics/novacron/backend/core/backup v0.0.0-20260926170121-980e0ba544b7
	github.com/khryptorgraphics/novacron/backend/core/orchestration v0.0.0-20250830173050-fe55263834f3
	github.com/lib/pq v1.10.9
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.11.1
	go.uber.org/zap v1.27.0
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/boombuler/barcode v1.0.1-0.20190219062509-6c824513bacc // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/chmduquesne/rollinghash v4.0.0+incompatible // indirect
	github.com/cilium/ebpf v0.22.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/golang-jwt/jwt/v5 v5.3.0 // indirect
	github.com/hashicorp/mdns v1.0.6 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/miekg/dns v1.1.55 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/nats-io/nats.go v1.37.0 // indirect
	github.com/nats-io/nkeys v0.4.9 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/pierrec/lz4/v4 v4.1.22 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/pquerna/otp v1.5.0 // indirect
	github.com/prometheus/client_golang v1.23.2 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.17.0 // indirect
	github.com/redis/go-redis/v9 v9.14.0 // indirect
	github.com/robfig/cron/v3 v3.0.1 // indirect
	github.com/skip2/go-qrcode v0.0.0-20200617195104-da1b6568686e // indirect
	github.com/stretchr/objx v0.5.2 // indirect
	github.com/vishvananda/netlink v1.3.1 // indirect
	github.com/vishvananda/netns v0.0.5 // indirect
	github.com/yalue/onnxruntime_go v1.22.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.yaml.in/yaml/v2 v2.4.2 // indirect
	golang.org/x/crypto v0.46.0 // indirect
	golang.org/x/mod v0.31.0 // indirect
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/sync v0.20.0 // indirect
	golang.org/x/sys v0.43.0 // indirect
	golang.org/x/time v0.12.0 // indirect
	golang.org/x/tools v0.40.1-0.20260108161641-ca281cf95054 // indirect
	google.golang.org/protobuf v1.36.10 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
