#!/bin/bash

# Deploy NovaCron Kubernetes Operator
set -e

NAMESPACE=${NAMESPACE:-novacron-system}
OPERATOR_IMAGE=${OPERATOR_IMAGE:-novacron/k8s-operator:latest}
NOVACRON_API_URL=${NOVACRON_API_URL:-http://novacron-api:8090}
NOVACRON_API_TOKEN=${NOVACRON_API_TOKEN}

echo "Deploying NovaCron Kubernetes Operator..."
echo "Namespace: $NAMESPACE"
echo "Image: $OPERATOR_IMAGE"
echo "API URL: $NOVACRON_API_URL"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply CRDs
echo "Applying Custom Resource Definitions..."
kubectl apply -f k8s-operator/deploy/crds/

# Create RBAC resources
echo "Creating RBAC resources..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: novacron-operator
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: novacron-operator
rules:
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch"]
- apiGroups: ["novacron.io"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: novacron-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: novacron-operator
subjects:
- kind: ServiceAccount
  name: novacron-operator
  namespace: $NAMESPACE
EOF

# Create API token secret if provided
if [ -n "$NOVACRON_API_TOKEN" ]; then
    echo "Creating API token secret..."
    kubectl create secret generic novacron-api-token \
        --from-literal=token="$NOVACRON_API_TOKEN" \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
fi

# Deploy operator
echo "Deploying operator..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-operator
  namespace: $NAMESPACE
  labels:
    app: novacron-operator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: novacron-operator
  template:
    metadata:
      labels:
        app: novacron-operator
    spec:
      serviceAccountName: novacron-operator
      containers:
      - name: operator
        image: $OPERATOR_IMAGE
        imagePullPolicy: Always
        args:
        - --novacron-api-url=$NOVACRON_API_URL
        - --concurrent-reconciles=3
        - --leader-elect=true
        env:
        - name: WATCH_NAMESPACE
          value: ""
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: OPERATOR_NAME
          value: novacron-operator
        volumeMounts:
        - name: api-token
          mountPath: /var/secrets/novacron
          readOnly: true
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: api-token
        secret:
          secretName: novacron-api-token
          optional: true
EOF

# Create monitoring service
echo "Creating monitoring service..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: novacron-operator-metrics
  namespace: $NAMESPACE
  labels:
    app: novacron-operator
spec:
  ports:
  - name: metrics
    port: 8080
    targetPort: 8080
  - name: health
    port: 8081
    targetPort: 8081
  selector:
    app: novacron-operator
EOF

# Wait for deployment to be ready
echo "Waiting for operator to be ready..."
kubectl wait --for=condition=available deployment/novacron-operator \
    --namespace=$NAMESPACE \
    --timeout=300s

echo "NovaCron Kubernetes Operator deployed successfully!"
echo ""
echo "Check status with:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -f deployment/novacron-operator -n $NAMESPACE"
echo ""
echo "Example usage:"
echo "  kubectl apply -f - <<EOF"
echo "  apiVersion: novacron.io/v1"
echo "  kind: VirtualMachine"
echo "  metadata:"
echo "    name: my-vm"
echo "    namespace: default"
echo "  spec:"
echo "    name: my-test-vm"
echo "    config:"
echo "      resources:"
echo "        cpu:"
echo "          request: \"1\""
echo "        memory:"
echo "          request: \"1Gi\""
echo "      image: ubuntu:20.04"
echo "      command: [\"/bin/bash\"]"
echo "      args: [\"-c\", \"while true; do sleep 30; done\"]"
echo "  EOF"