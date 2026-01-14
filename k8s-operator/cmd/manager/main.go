package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/controllers"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/novacron"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/providers"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/ai"
	"github.com/khryptorgraphics/novacron/k8s-operator/pkg/cache"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(novacronv1.AddToScheme(scheme))
}

func main() {
	var (
		metricsAddr              string
		enableLeaderElection     bool
		probeAddr               string
		novacronApiUrl          string
		novacronApiToken        string
		concurrentReconciles    int
	)

	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&novacronApiUrl, "novacron-api-url", "http://localhost:8090", "NovaCron API server URL")
	flag.StringVar(&novacronApiToken, "novacron-api-token", "", "NovaCron API authentication token")
	flag.IntVar(&concurrentReconciles, "concurrent-reconciles", 1, "Number of concurrent reconciles")

	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// Print version information
	setupLog.Info("Starting NovaCron Enhanced Operator",
		"version", getVersion(),
		"go-version", runtime.Version(),
		"go-os", runtime.GOOS,
		"go-arch", runtime.GOARCH,
	)

	// Create manager
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     metricsAddr,
		Port:                   9443,
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "novacron-operator-leader",
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// Initialize NovaCron client
	novacronClient, err := novacron.NewClient(novacron.ClientConfig{
		BaseURL: novacronApiUrl,
		Token:   novacronApiToken,
	})
	if err != nil {
		setupLog.Error(err, "unable to create NovaCron client")
		os.Exit(1)
	}

	// Initialize cloud provider manager
	cloudProviderManager := providers.NewCloudProviderManager()
	cloudProviderFactory := providers.NewCloudProviderFactory()

	// Register mock provider for development
	mockProvider, err := cloudProviderFactory.Create(providers.ProviderConfig{
		Name: "mock",
		Type: "mock",
	})
	if err != nil {
		setupLog.Error(err, "unable to create mock provider")
		os.Exit(1)
	}
	if err := cloudProviderManager.RegisterProvider(mockProvider); err != nil {
		setupLog.Error(err, "unable to register mock provider")
		os.Exit(1)
	}

	// Initialize AI scheduling engine
	aiEngine := ai.NewMockSchedulingEngine()

	// Initialize cache manager
	cacheManager := cache.NewMockManager()

	// Set up VirtualMachine controller
	if err = (&controllers.VirtualMachineReconciler{
		Client:         mgr.GetClient(),
		Scheme:         mgr.GetScheme(),
		NovaCronClient: novacronClient,
		Recorder:       mgr.GetEventRecorderFor("vm-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "VirtualMachine")
		os.Exit(1)
	}

	// Set up VMTemplate controller
	if err = (&controllers.VMTemplateReconciler{
		Client:         mgr.GetClient(),
		Scheme:         mgr.GetScheme(),
		NovaCronClient: novacronClient,
		Recorder:       mgr.GetEventRecorderFor("vmtemplate-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "VMTemplate")
		os.Exit(1)
	}

	// Set up VMCluster controller
	if err = (&controllers.VMClusterReconciler{
		Client:         mgr.GetClient(),
		Scheme:         mgr.GetScheme(),
		NovaCronClient: novacronClient,
		Recorder:       mgr.GetEventRecorderFor("vmcluster-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "VMCluster")
		os.Exit(1)
	}

	// Set up MultiCloudVM controller
	if err = (&controllers.MultiCloudVMReconciler{
		Client:         mgr.GetClient(),
		Scheme:         mgr.GetScheme(),
		NovaCronClient: novacronClient,
		CloudProviders: cloudProviderManager,
		Recorder:       mgr.GetEventRecorderFor("multicloudvm-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "MultiCloudVM")
		os.Exit(1)
	}

	// Set up AISchedulingPolicy controller
	if err = (&controllers.AISchedulingPolicyReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		AIEngine: aiEngine,
		Recorder: mgr.GetEventRecorderFor("aischedulingpolicy-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "AISchedulingPolicy")
		os.Exit(1)
	}

	// Set up CacheIntegration controller
	if err = (&controllers.CacheIntegrationReconciler{
		Client:       mgr.GetClient(),
		Scheme:       mgr.GetScheme(),
		CacheManager: cacheManager,
		Recorder:     mgr.GetEventRecorderFor("cacheintegration-controller"),
	}).SetupWithManager(mgr, concurrentReconciles); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "CacheIntegration")
		os.Exit(1)
	}

	// Add health checks
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting enhanced NovaCron operator with multi-cloud and AI capabilities")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func getVersion() string {
	return "v2.0.0-enhanced"
}
