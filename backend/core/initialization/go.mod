module novacron/backend/core/initialization

go 1.21

require (
	gopkg.in/yaml.v3 v3.0.1
)

replace novacron/backend/core/initialization/config => ./config
replace novacron/backend/core/initialization/orchestrator => ./orchestrator
replace novacron/backend/core/initialization/di => ./di
replace novacron/backend/core/initialization/recovery => ./recovery
replace novacron/backend/core/initialization/logger => ./logger
