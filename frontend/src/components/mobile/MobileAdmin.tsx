// Mobile-first administration interface for NovaCron
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Modal,
  Alert,
  Platform,
  Dimensions,
  RefreshControl,
  FlatList,
  Switch,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LineChart, PieChart, BarChart } from 'react-native-chart-kit';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer } from '@react-navigation/native';
import NetInfo from '@react-native-community/netinfo';
import PushNotification from 'react-native-push-notification';
import FingerprintScanner from 'react-native-fingerprint-scanner';
import Voice from '@react-native-voice/voice';
import Haptics from 'react-native-haptic-feedback';

// API Service
class MobileAPIService {
  private baseURL: string;
  private token: string | null = null;
  private wsConnection: WebSocket | null = null;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  async authenticate(credentials: AuthCredentials): Promise<AuthResponse> {
    const response = await fetch(`${this.baseURL}/auth/mobile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });

    const data = await response.json();
    if (data.token) {
      this.token = data.token;
      await AsyncStorage.setItem('auth_token', data.token);
    }
    return data;
  }

  async biometricAuth(): Promise<boolean> {
    try {
      await FingerprintScanner.authenticate({
        title: 'Authenticate',
        subTitle: 'Access NovaCron Admin',
        description: 'Touch the fingerprint sensor',
        fallbackEnabled: true,
      });
      return true;
    } catch (error) {
      return false;
    }
  }

  connectWebSocket(onMessage: (data: any) => void) {
    this.wsConnection = new WebSocket(`${this.baseURL.replace('http', 'ws')}/ws`);
    
    this.wsConnection.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  async makeRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
    const headers = {
      ...options.headers,
      Authorization: `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };

    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    return response.json();
  }
}

// Types
interface VM {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'migrating';
  cpu: number;
  memory: number;
  disk: number;
  location: string;
  health: 'healthy' | 'warning' | 'critical';
}

interface Metric {
  timestamp: number;
  value: number;
  label: string;
}

interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

interface AuthCredentials {
  username?: string;
  password?: string;
  biometric?: boolean;
}

interface AuthResponse {
  token: string;
  user: {
    id: string;
    name: string;
    role: string;
    permissions: string[];
  };
}

// Custom Hooks
const useOfflineSync = () => {
  const [isOffline, setIsOffline] = useState(false);
  const [syncQueue, setSyncQueue] = useState<any[]>([]);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsOffline(!state.isConnected);
      
      if (state.isConnected && syncQueue.length > 0) {
        // Sync queued operations
        syncQueue.forEach(async (operation) => {
          try {
            await operation();
          } catch (error) {
            console.error('Sync error:', error);
          }
        });
        setSyncQueue([]);
      }
    });

    return () => unsubscribe();
  }, [syncQueue]);

  const queueOperation = useCallback((operation: () => Promise<any>) => {
    if (isOffline) {
      setSyncQueue(prev => [...prev, operation]);
      return Promise.resolve({ queued: true });
    }
    return operation();
  }, [isOffline]);

  return { isOffline, queueOperation, pendingSync: syncQueue.length };
};

const useVoiceCommands = (onCommand: (command: string) => void) => {
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    Voice.onSpeechResults = (e: any) => {
      if (e.value && e.value.length > 0) {
        onCommand(e.value[0]);
      }
    };

    Voice.onSpeechError = (e: any) => {
      console.error('Voice error:', e);
      setIsListening(false);
    };

    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, [onCommand]);

  const startListening = async () => {
    try {
      await Voice.start('en-US');
      setIsListening(true);
    } catch (error) {
      console.error('Voice start error:', error);
    }
  };

  const stopListening = async () => {
    try {
      await Voice.stop();
      setIsListening(false);
    } catch (error) {
      console.error('Voice stop error:', error);
    }
  };

  return { isListening, startListening, stopListening };
};

// Components
const MobileAdminDashboard: React.FC = () => {
  const queryClient = useQueryClient();
  const [selectedVM, setSelectedVM] = useState<VM | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const { isOffline, queueOperation, pendingSync } = useOfflineSync();
  
  // Fetch VMs
  const { data: vms, refetch: refetchVMs } = useQuery(
    'vms',
    () => apiService.makeRequest('/api/vms'),
    {
      refetchInterval: isOffline ? false : 30000,
      staleTime: 60000,
    }
  );

  // Fetch metrics
  const { data: metrics } = useQuery(
    'metrics',
    () => apiService.makeRequest('/api/metrics'),
    {
      refetchInterval: isOffline ? false : 10000,
    }
  );

  // Fetch alerts
  const { data: alerts } = useQuery(
    'alerts',
    () => apiService.makeRequest('/api/alerts'),
    {
      refetchInterval: isOffline ? false : 5000,
    }
  );

  // Voice commands
  const handleVoiceCommand = useCallback(async (command: string) => {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('start') && lowerCommand.includes('vm')) {
      const vmName = extractVMName(command);
      await startVM(vmName);
    } else if (lowerCommand.includes('stop') && lowerCommand.includes('vm')) {
      const vmName = extractVMName(command);
      await stopVM(vmName);
    } else if (lowerCommand.includes('show') && lowerCommand.includes('metrics')) {
      // Navigate to metrics view
    } else if (lowerCommand.includes('migrate')) {
      const vmName = extractVMName(command);
      await migrateVM(vmName);
    }
    
    Haptics.trigger('impactLight');
  }, []);

  const { isListening, startListening, stopListening } = useVoiceCommands(handleVoiceCommand);

  // VM operations
  const startVMMutation = useMutation(
    (vmId: string) => apiService.makeRequest(`/api/vms/${vmId}/start`, { method: 'POST' }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vms');
        showNotification('VM Started', 'success');
      },
    }
  );

  const stopVMMutation = useMutation(
    (vmId: string) => apiService.makeRequest(`/api/vms/${vmId}/stop`, { method: 'POST' }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vms');
        showNotification('VM Stopped', 'success');
      },
    }
  );

  const migrateVMMutation = useMutation(
    ({ vmId, destination }: { vmId: string; destination: string }) =>
      apiService.makeRequest(`/api/vms/${vmId}/migrate`, {
        method: 'POST',
        body: JSON.stringify({ destination }),
      }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vms');
        showNotification('Migration Started', 'info');
      },
    }
  );

  // Refresh handler
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await Promise.all([
      refetchVMs(),
      queryClient.invalidateQueries('metrics'),
      queryClient.invalidateQueries('alerts'),
    ]);
    setRefreshing(false);
  }, [refetchVMs, queryClient]);

  // Chart configuration
  const chartConfig = {
    backgroundGradientFrom: '#1E2923',
    backgroundGradientFromOpacity: 0,
    backgroundGradientTo: '#08130D',
    backgroundGradientToOpacity: 0.5,
    color: (opacity = 1) => `rgba(26, 255, 146, ${opacity})`,
    strokeWidth: 2,
    barPercentage: 0.5,
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1a1a1a" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>NovaCron Admin</Text>
        <View style={styles.headerActions}>
          {isOffline && (
            <View style={styles.offlineIndicator}>
              <Icon name="cloud-off" size={20} color="#ff6b6b" />
              <Text style={styles.offlineText}>Offline ({pendingSync})</Text>
            </View>
          )}
          <TouchableOpacity
            style={[styles.voiceButton, isListening && styles.voiceButtonActive]}
            onPress={isListening ? stopListening : startListening}
          >
            <Icon name={isListening ? 'mic' : 'mic-none'} size={24} color="#fff" />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView
        style={styles.content}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Alert Banner */}
        {alerts && alerts.filter((a: Alert) => !a.acknowledged).length > 0 && (
          <View style={styles.alertBanner}>
            <Icon name="warning" size={20} color="#ff6b6b" />
            <Text style={styles.alertText}>
              {alerts.filter((a: Alert) => !a.acknowledged).length} active alerts
            </Text>
            <TouchableOpacity onPress={() => navigateToAlerts()}>
              <Text style={styles.alertAction}>View</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Quick Stats */}
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>{vms?.length || 0}</Text>
            <Text style={styles.statLabel}>Total VMs</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>
              {vms?.filter((vm: VM) => vm.status === 'running').length || 0}
            </Text>
            <Text style={styles.statLabel}>Running</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>
              {metrics?.cpu?.current?.toFixed(1) || 0}%
            </Text>
            <Text style={styles.statLabel}>CPU Usage</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>
              {metrics?.memory?.current?.toFixed(1) || 0}%
            </Text>
            <Text style={styles.statLabel}>Memory</Text>
          </View>
        </View>

        {/* Performance Chart */}
        {metrics?.history && (
          <View style={styles.chartContainer}>
            <Text style={styles.sectionTitle}>Performance Trends</Text>
            <LineChart
              data={{
                labels: metrics.history.labels,
                datasets: [
                  {
                    data: metrics.history.cpu,
                    color: (opacity = 1) => `rgba(134, 65, 244, ${opacity})`,
                  },
                  {
                    data: metrics.history.memory,
                    color: (opacity = 1) => `rgba(255, 159, 64, ${opacity})`,
                  },
                ],
              }}
              width={Dimensions.get('window').width - 32}
              height={200}
              chartConfig={chartConfig}
              bezier
              style={styles.chart}
            />
          </View>
        )}

        {/* VM List */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Virtual Machines</Text>
          <FlatList
            data={vms || []}
            keyExtractor={(item) => item.id}
            scrollEnabled={false}
            renderItem={({ item }: { item: VM }) => (
              <TouchableOpacity
                style={styles.vmCard}
                onPress={() => setSelectedVM(item)}
                onLongPress={() => showVMActions(item)}
              >
                <View style={styles.vmHeader}>
                  <View style={styles.vmInfo}>
                    <Text style={styles.vmName}>{item.name}</Text>
                    <Text style={styles.vmLocation}>{item.location}</Text>
                  </View>
                  <View style={[styles.vmStatus, styles[`status_${item.status}`]]}>
                    <Text style={styles.vmStatusText}>{item.status}</Text>
                  </View>
                </View>
                <View style={styles.vmMetrics}>
                  <View style={styles.vmMetric}>
                    <Icon name="memory" size={16} color="#666" />
                    <Text style={styles.vmMetricValue}>CPU: {item.cpu}%</Text>
                  </View>
                  <View style={styles.vmMetric}>
                    <Icon name="storage" size={16} color="#666" />
                    <Text style={styles.vmMetricValue}>RAM: {item.memory}GB</Text>
                  </View>
                  <View style={styles.vmMetric}>
                    <Icon name="disc-full" size={16} color="#666" />
                    <Text style={styles.vmMetricValue}>Disk: {item.disk}GB</Text>
                  </View>
                </View>
                <View style={styles.vmActions}>
                  <TouchableOpacity
                    style={[styles.vmActionButton, item.status === 'running' && styles.vmActionButtonDanger]}
                    onPress={() => item.status === 'running' ? stopVM(item.id) : startVM(item.id)}
                  >
                    <Icon 
                      name={item.status === 'running' ? 'stop' : 'play-arrow'} 
                      size={20} 
                      color="#fff" 
                    />
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.vmActionButton}
                    onPress={() => navigateToVMDetails(item)}
                  >
                    <Icon name="info" size={20} color="#fff" />
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.vmActionButton}
                    onPress={() => showMigrateDialog(item)}
                  >
                    <Icon name="swap-horiz" size={20} color="#fff" />
                  </TouchableOpacity>
                </View>
              </TouchableOpacity>
            )}
          />
        </View>

        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <TouchableOpacity style={styles.quickActionButton} onPress={createNewVM}>
            <Icon name="add-circle" size={32} color="#4CAF50" />
            <Text style={styles.quickActionLabel}>New VM</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickActionButton} onPress={runHealthCheck}>
            <Icon name="health-and-safety" size={32} color="#2196F3" />
            <Text style={styles.quickActionLabel}>Health Check</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickActionButton} onPress={optimizeResources}>
            <Icon name="auto-fix-high" size={32} color="#FF9800" />
            <Text style={styles.quickActionLabel}>Optimize</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.quickActionButton} onPress={viewReports}>
            <Icon name="analytics" size={32} color="#9C27B0" />
            <Text style={styles.quickActionLabel}>Reports</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Floating Action Button */}
      <TouchableOpacity style={styles.fab} onPress={showCommandPalette}>
        <Icon name="terminal" size={24} color="#fff" />
      </TouchableOpacity>
    </SafeAreaView>
  );
};

// Alert Screen
const AlertScreen: React.FC = () => {
  const { data: alerts, refetch } = useQuery('alerts', () =>
    apiService.makeRequest('/api/alerts')
  );

  const acknowledgeMutation = useMutation(
    (alertId: string) =>
      apiService.makeRequest(`/api/alerts/${alertId}/acknowledge`, {
        method: 'POST',
      }),
    {
      onSuccess: () => refetch(),
    }
  );

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#ff4444';
      case 'warning':
        return '#ffaa00';
      default:
        return '#44ff44';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Alerts</Text>
      </View>

      <FlatList
        data={alerts || []}
        keyExtractor={(item) => item.id}
        renderItem={({ item }: { item: Alert }) => (
          <View style={[styles.alertCard, { borderLeftColor: getSeverityColor(item.severity) }]}>
            <View style={styles.alertHeader}>
              <View style={styles.alertSeverity}>
                <Icon 
                  name={item.severity === 'critical' ? 'error' : 'warning'} 
                  size={24} 
                  color={getSeverityColor(item.severity)} 
                />
                <Text style={[styles.alertSeverityText, { color: getSeverityColor(item.severity) }]}>
                  {item.severity.toUpperCase()}
                </Text>
              </View>
              <Text style={styles.alertTime}>
                {new Date(item.timestamp).toLocaleTimeString()}
              </Text>
            </View>
            <Text style={styles.alertMessage}>{item.message}</Text>
            {!item.acknowledged && (
              <TouchableOpacity
                style={styles.acknowledgeButton}
                onPress={() => acknowledgeMutation.mutate(item.id)}
              >
                <Text style={styles.acknowledgeButtonText}>Acknowledge</Text>
              </TouchableOpacity>
            )}
          </View>
        )}
      />
    </SafeAreaView>
  );
};

// Settings Screen
const SettingsScreen: React.FC = () => {
  const [biometricEnabled, setBiometricEnabled] = useState(false);
  const [pushEnabled, setPushEnabled] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('30');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const settings = await AsyncStorage.getItem('settings');
    if (settings) {
      const parsed = JSON.parse(settings);
      setBiometricEnabled(parsed.biometric || false);
      setPushEnabled(parsed.push || true);
      setAutoRefresh(parsed.autoRefresh || true);
      setRefreshInterval(parsed.refreshInterval || '30');
    }
  };

  const saveSettings = async () => {
    const settings = {
      biometric: biometricEnabled,
      push: pushEnabled,
      autoRefresh,
      refreshInterval,
    };
    await AsyncStorage.setItem('settings', JSON.stringify(settings));
    Alert.alert('Success', 'Settings saved successfully');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.content}>
        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Security</Text>
          
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Biometric Authentication</Text>
            <Switch
              value={biometricEnabled}
              onValueChange={setBiometricEnabled}
              trackColor={{ false: '#767577', true: '#4CAF50' }}
            />
          </View>
        </View>

        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Notifications</Text>
          
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Push Notifications</Text>
            <Switch
              value={pushEnabled}
              onValueChange={setPushEnabled}
              trackColor={{ false: '#767577', true: '#4CAF50' }}
            />
          </View>
        </View>

        <View style={styles.settingsSection}>
          <Text style={styles.settingsSectionTitle}>Data & Sync</Text>
          
          <View style={styles.settingRow}>
            <Text style={styles.settingLabel}>Auto Refresh</Text>
            <Switch
              value={autoRefresh}
              onValueChange={setAutoRefresh}
              trackColor={{ false: '#767577', true: '#4CAF50' }}
            />
          </View>
          
          {autoRefresh && (
            <View style={styles.settingRow}>
              <Text style={styles.settingLabel}>Refresh Interval (seconds)</Text>
              <TextInput
                style={styles.settingInput}
                value={refreshInterval}
                onChangeText={setRefreshInterval}
                keyboardType="numeric"
                placeholder="30"
              />
            </View>
          )}
        </View>

        <TouchableOpacity style={styles.saveButton} onPress={saveSettings}>
          <Text style={styles.saveButtonText}>Save Settings</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#2a2a2a',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  offlineIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
  },
  offlineText: {
    color: '#ff6b6b',
    marginLeft: 4,
  },
  voiceButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: '#333',
  },
  voiceButtonActive: {
    backgroundColor: '#4CAF50',
  },
  content: {
    flex: 1,
  },
  alertBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: '#ff6b6b22',
    margin: 16,
    borderRadius: 8,
  },
  alertText: {
    flex: 1,
    color: '#ff6b6b',
    marginLeft: 8,
  },
  alertAction: {
    color: '#fff',
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 8,
  },
  statCard: {
    width: '50%',
    padding: 8,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  chartContainer: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  section: {
    padding: 16,
  },
  vmCard: {
    backgroundColor: '#2a2a2a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  vmHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  vmInfo: {
    flex: 1,
  },
  vmName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
  vmLocation: {
    fontSize: 12,
    color: '#999',
    marginTop: 2,
  },
  vmStatus: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  status_running: {
    backgroundColor: '#4CAF5022',
  },
  status_stopped: {
    backgroundColor: '#ff444422',
  },
  status_migrating: {
    backgroundColor: '#2196F322',
  },
  vmStatusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  vmMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  vmMetric: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  vmMetricValue: {
    color: '#999',
    fontSize: 12,
    marginLeft: 4,
  },
  vmActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  vmActionButton: {
    backgroundColor: '#333',
    padding: 8,
    borderRadius: 8,
    minWidth: 40,
    alignItems: 'center',
  },
  vmActionButtonDanger: {
    backgroundColor: '#ff4444',
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 16,
  },
  quickActionButton: {
    alignItems: 'center',
    padding: 12,
  },
  quickActionLabel: {
    color: '#999',
    fontSize: 12,
    marginTop: 4,
  },
  fab: {
    position: 'absolute',
    bottom: 20,
    right: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  alertCard: {
    backgroundColor: '#2a2a2a',
    margin: 16,
    marginBottom: 8,
    padding: 16,
    borderRadius: 8,
    borderLeftWidth: 4,
  },
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  alertSeverity: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  alertSeverityText: {
    fontWeight: 'bold',
    marginLeft: 8,
  },
  alertTime: {
    color: '#999',
    fontSize: 12,
  },
  alertMessage: {
    color: '#fff',
    marginBottom: 12,
  },
  acknowledgeButton: {
    backgroundColor: '#4CAF50',
    padding: 8,
    borderRadius: 4,
    alignItems: 'center',
  },
  acknowledgeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  settingsSection: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  settingsSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  settingLabel: {
    color: '#fff',
    fontSize: 14,
  },
  settingInput: {
    backgroundColor: '#333',
    color: '#fff',
    padding: 8,
    borderRadius: 4,
    width: 80,
    textAlign: 'center',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    margin: 16,
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});

// API Service instance
const apiService = new MobileAPIService('http://localhost:8090');

// Helper functions
function extractVMName(command: string): string {
  const words = command.split(' ');
  const vmIndex = words.findIndex(w => w.toLowerCase() === 'vm');
  if (vmIndex >= 0 && vmIndex < words.length - 1) {
    return words[vmIndex + 1];
  }
  return '';
}

async function startVM(vmId: string) {
  try {
    await apiService.makeRequest(`/api/vms/${vmId}/start`, { method: 'POST' });
    showNotification('VM Started', 'success');
  } catch (error) {
    showNotification('Failed to start VM', 'error');
  }
}

async function stopVM(vmId: string) {
  try {
    await apiService.makeRequest(`/api/vms/${vmId}/stop`, { method: 'POST' });
    showNotification('VM Stopped', 'success');
  } catch (error) {
    showNotification('Failed to stop VM', 'error');
  }
}

async function migrateVM(vmId: string) {
  // Implementation would show migration dialog
  console.log('Migrate VM:', vmId);
}

function showNotification(message: string, type: 'success' | 'error' | 'info') {
  PushNotification.localNotification({
    title: 'NovaCron',
    message,
    playSound: true,
    soundName: 'default',
  });
}

function showVMActions(vm: VM) {
  Alert.alert(
    vm.name,
    'Choose an action',
    [
      { text: 'Start/Stop', onPress: () => vm.status === 'running' ? stopVM(vm.id) : startVM(vm.id) },
      { text: 'Migrate', onPress: () => migrateVM(vm.id) },
      { text: 'Delete', onPress: () => deleteVM(vm.id), style: 'destructive' },
      { text: 'Cancel', style: 'cancel' },
    ]
  );
}

async function deleteVM(vmId: string) {
  Alert.alert(
    'Confirm Delete',
    'Are you sure you want to delete this VM?',
    [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            await apiService.makeRequest(`/api/vms/${vmId}`, { method: 'DELETE' });
            showNotification('VM Deleted', 'success');
          } catch (error) {
            showNotification('Failed to delete VM', 'error');
          }
        },
      },
    ]
  );
}

function navigateToVMDetails(vm: VM) {
  // Navigation implementation
  console.log('Navigate to VM details:', vm.id);
}

function navigateToAlerts() {
  // Navigation implementation
  console.log('Navigate to alerts');
}

function showMigrateDialog(vm: VM) {
  // Show migration dialog
  console.log('Show migrate dialog for:', vm.id);
}

function createNewVM() {
  // Show VM creation dialog
  console.log('Create new VM');
}

function runHealthCheck() {
  // Run system health check
  console.log('Run health check');
}

function optimizeResources() {
  // Trigger resource optimization
  console.log('Optimize resources');
}

function viewReports() {
  // Navigate to reports
  console.log('View reports');
}

function showCommandPalette() {
  // Show command palette
  console.log('Show command palette');
}

// Navigation Setup
const Tab = createBottomTabNavigator();

export default function MobileApp() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          tabBarStyle: {
            backgroundColor: '#2a2a2a',
            borderTopColor: '#333',
          },
          tabBarActiveTintColor: '#4CAF50',
          tabBarInactiveTintColor: '#999',
          headerShown: false,
        }}
      >
        <Tab.Screen
          name="Dashboard"
          component={MobileAdminDashboard}
          options={{
            tabBarIcon: ({ color, size }) => (
              <Icon name="dashboard" size={size} color={color} />
            ),
          }}
        />
        <Tab.Screen
          name="Alerts"
          component={AlertScreen}
          options={{
            tabBarIcon: ({ color, size }) => (
              <Icon name="notifications" size={size} color={color} />
            ),
          }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{
            tabBarIcon: ({ color, size }) => (
              <Icon name="settings" size={size} color={color} />
            ),
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}