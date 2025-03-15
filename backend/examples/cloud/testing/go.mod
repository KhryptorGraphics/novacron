module github.com/novacron/backend/examples/cloud/testing

go 1.20

require (
	github.com/novacron/backend v0.0.0
	github.com/novacron/backend/examples/cloud/testsuite v0.0.0
)

replace (
	github.com/novacron/backend => ../../../..
	github.com/novacron/backend/examples/cloud/testsuite => ../testsuite
)
