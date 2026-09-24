# NovaCron Production Deployment Guide

## Canonical entry point

The only supported production path is documented in [../deploy/README.md](../deploy/README.md).
It covers systemd units (verified correct), install-node.sh, health checks, and
migration sequencing.

This file exists because several older references still point here. All
previous content in this document referenced files that do not exist in the
repository (e.g. deployment/kubernetes/backend-deployment.yaml,
scripts/deploy-edge-agent.sh, or a /ready probe) and has been removed to avoid
misleading operators.
