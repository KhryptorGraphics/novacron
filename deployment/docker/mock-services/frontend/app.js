class NovaCronApp {
    constructor() {
        this.token = localStorage.getItem('novacron_token');
        this.user = JSON.parse(localStorage.getItem('novacron_user') || 'null');
        this.socket = null;
        this.chart = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateUI();
        
        if (this.token) {
            this.connectWebSocket();
            this.loadDashboard();
        }
    }
    
    setupEventListeners() {
        // Login/Logout
        document.getElementById('login-btn').addEventListener('click', () => this.showLoginModal());
        document.getElementById('logout-btn').addEventListener('click', () => this.logout());
        document.getElementById('login-form').addEventListener('submit', (e) => this.handleLogin(e));
        
        // VM Management
        document.getElementById('create-vm-btn').addEventListener('click', () => this.showCreateVMModal());
        document.getElementById('create-vm-form').addEventListener('submit', (e) => this.handleCreateVM(e));
        document.getElementById('cancel-vm-btn').addEventListener('click', () => this.hideCreateVMModal());
        
        // Close modals on outside click
        document.getElementById('login-modal').addEventListener('click', (e) => {
            if (e.target.id === 'login-modal') this.hideLoginModal();
        });
        
        document.getElementById('create-vm-modal').addEventListener('click', (e) => {
            if (e.target.id === 'create-vm-modal') this.hideCreateVMModal();
        });
    }
    
    updateUI() {
        const loginBtn = document.getElementById('login-btn');
        const logoutBtn = document.getElementById('logout-btn');
        const mainContent = document.getElementById('main-content');
        const loginModal = document.getElementById('login-modal');
        const userInfo = document.getElementById('user-info');
        
        if (this.token && this.user) {
            loginBtn.classList.add('hidden');
            logoutBtn.classList.remove('hidden');
            mainContent.classList.remove('hidden');
            loginModal.classList.add('hidden');
            userInfo.textContent = `${this.user.username} (${this.user.role})`;
        } else {
            loginBtn.classList.remove('hidden');
            logoutBtn.classList.add('hidden');
            mainContent.classList.add('hidden');
            loginModal.classList.remove('hidden');
            userInfo.textContent = 'Guest';
        }
    }
    
    async apiCall(endpoint, options = {}) {
        const url = `http://localhost:15561${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...(this.token ? { 'Authorization': `Bearer ${this.token}` } : {}),
            ...options.headers
        };
        
        try {
            const response = await fetch(url, {
                ...options,
                headers
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'API request failed');
            }
            
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            if (error.message.includes('token') || error.message.includes('401')) {
                this.logout();
            }
            throw error;
        }
    }
    
    async handleLogin(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        try {
            const response = await this.apiCall('/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });
            
            this.token = response.token;
            this.user = response.user;
            
            localStorage.setItem('novacron_token', this.token);
            localStorage.setItem('novacron_user', JSON.stringify(this.user));
            
            this.updateUI();
            this.connectWebSocket();
            this.loadDashboard();
            
        } catch (error) {
            alert('Login failed: ' + error.message);
        }
    }
    
    logout() {
        this.token = null;
        this.user = null;
        
        localStorage.removeItem('novacron_token');
        localStorage.removeItem('novacron_user');
        
        if (this.socket) {
            this.socket.close();
        }
        
        this.updateUI();
    }
    
    connectWebSocket() {
        try {
            this.socket = new WebSocket('ws://localhost:15561');
            
            this.socket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('connected');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                
                // Reconnect after 5 seconds
                setTimeout(() => {
                    if (this.token) {
                        this.connectWebSocket();
                    }
                }, 5000);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error');
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus('error');
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'system_metrics':
                this.updateSystemMetrics(data.data);
                break;
            case 'vm_status_change':
                this.handleVMStatusChange(data.data);
                break;
            case 'vm_deleted':
                this.handleVMDeleted(data.data);
                break;
            default:
                console.log('Unknown WebSocket message:', data);
        }
    }
    
    updateConnectionStatus(status) {
        const indicator = document.getElementById('status-indicator');
        const icon = indicator.querySelector('i');
        const text = indicator.querySelector('span');
        
        switch (status) {
            case 'connected':
                icon.className = 'fas fa-circle text-green-500 mr-2';
                text.textContent = 'Connected';
                break;
            case 'disconnected':
                icon.className = 'fas fa-circle text-yellow-500 mr-2';
                text.textContent = 'Reconnecting...';
                break;
            case 'error':
                icon.className = 'fas fa-circle text-red-500 mr-2';
                text.textContent = 'Connection Error';
                break;
        }
    }
    
    async loadDashboard() {
        try {
            const [stats, vms] = await Promise.all([
                this.apiCall('/api/dashboard/stats'),
                this.apiCall('/api/vms')
            ]);
            
            this.updateDashboardStats(stats);
            this.updateVMTable(vms.vms);
            this.createVMStatusChart(stats.vms);
            
        } catch (error) {
            console.error('Failed to load dashboard:', error);
            alert('Failed to load dashboard data');
        }
    }
    
    updateDashboardStats(stats) {
        document.getElementById('total-vms').textContent = stats.vms.total;
        document.getElementById('running-vms').textContent = stats.vms.running;
        document.getElementById('stopped-vms').textContent = stats.vms.stopped;
        document.getElementById('total-users').textContent = stats.users.total;
        
        this.updateSystemMetrics(stats.system);
    }
    
    updateSystemMetrics(metrics) {
        const cpu = Math.round(metrics.cpu);
        const memory = Math.round(metrics.memory);
        const disk = Math.round(metrics.disk);
        
        document.getElementById('cpu-usage').textContent = `${cpu}%`;
        document.getElementById('memory-usage').textContent = `${memory}%`;
        document.getElementById('disk-usage').textContent = `${disk}%`;
        
        document.getElementById('cpu-bar').style.width = `${cpu}%`;
        document.getElementById('memory-bar').style.width = `${memory}%`;
        document.getElementById('disk-bar').style.width = `${disk}%`;
    }
    
    createVMStatusChart(vmStats) {
        const ctx = document.getElementById('vm-status-chart').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }
        
        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Running', 'Stopped', 'Paused', 'Error'],
                datasets: [{
                    data: [vmStats.running, vmStats.stopped, vmStats.paused, vmStats.error],
                    backgroundColor: [
                        '#10B981',
                        '#F59E0B',
                        '#8B5CF6',
                        '#EF4444'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    updateVMTable(vms) {
        const tbody = document.getElementById('vm-table-body');
        tbody.innerHTML = '';
        
        vms.forEach(vm => {
            const row = this.createVMRow(vm);
            tbody.appendChild(row);
        });
    }
    
    createVMRow(vm) {
        const row = document.createElement('tr');
        row.dataset.vmId = vm.id;
        
        const statusColors = {
            running: 'text-green-600',
            stopped: 'text-gray-600',
            paused: 'text-yellow-600',
            error: 'text-red-600',
            creating: 'text-blue-600'
        };
        
        row.innerHTML = `
            <td class=\"px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900\">${vm.name}</td>
            <td class=\"px-6 py-4 whitespace-nowrap\">
                <span class=\"px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 ${statusColors[vm.status]}\">${vm.status}</span>
            </td>
            <td class=\"px-6 py-4 whitespace-nowrap text-sm text-gray-500\">${vm.cpu_cores} cores</td>
            <td class=\"px-6 py-4 whitespace-nowrap text-sm text-gray-500\">${Math.round(vm.memory_mb / 1024)} GB</td>
            <td class=\"px-6 py-4 whitespace-nowrap text-sm text-gray-500\">${vm.host_node || 'N/A'}</td>
            <td class=\"px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2\">
                ${vm.status === 'stopped' ? 
                    `<button onclick=\"app.startVM('${vm.id}')\" class=\"text-green-600 hover:text-green-900\">
                        <i class=\"fas fa-play\"></i> Start
                    </button>` : ''}
                ${vm.status === 'running' ? 
                    `<button onclick=\"app.stopVM('${vm.id}')\" class=\"text-red-600 hover:text-red-900\">
                        <i class=\"fas fa-stop\"></i> Stop
                    </button>` : ''}
                <button onclick=\"app.deleteVM('${vm.id}')\" class=\"text-red-600 hover:text-red-900 ml-2\">
                    <i class=\"fas fa-trash\"></i> Delete
                </button>
            </td>
        `;
        
        return row;
    }
    
    handleVMStatusChange(data) {
        const row = document.querySelector(`tr[data-vm-id=\"${data.vm_id}\"]`);
        if (row) {
            const statusCell = row.children[1].querySelector('span');
            statusCell.textContent = data.status;
            
            const statusColors = {
                running: 'text-green-600',
                stopped: 'text-gray-600',
                paused: 'text-yellow-600',
                error: 'text-red-600'
            };
            
            statusCell.className = `px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 ${statusColors[data.status]}`;
            
            // Update action buttons
            const actionsCell = row.children[5];
            const vm = { id: data.vm_id, status: data.status };
            actionsCell.innerHTML = `
                ${data.status === 'stopped' ? 
                    `<button onclick=\"app.startVM('${vm.id}')\" class=\"text-green-600 hover:text-green-900\">
                        <i class=\"fas fa-play\"></i> Start
                    </button>` : ''}
                ${data.status === 'running' ? 
                    `<button onclick=\"app.stopVM('${vm.id}')\" class=\"text-red-600 hover:text-red-900\">
                        <i class=\"fas fa-stop\"></i> Stop
                    </button>` : ''}
                <button onclick=\"app.deleteVM('${vm.id}')\" class=\"text-red-600 hover:text-red-900 ml-2\">
                    <i class=\"fas fa-trash\"></i> Delete
                </button>
            `;
        }
        
        // Refresh stats
        this.loadDashboard();
    }
    
    handleVMDeleted(data) {
        const row = document.querySelector(`tr[data-vm-id=\"${data.vm_id}\"]`);
        if (row) {
            row.remove();
        }
        
        // Refresh stats
        this.loadDashboard();
    }
    
    async startVM(vmId) {
        try {
            await this.apiCall(`/api/vms/${vmId}/start`, { method: 'POST' });
        } catch (error) {
            alert('Failed to start VM: ' + error.message);
        }
    }
    
    async stopVM(vmId) {
        try {
            await this.apiCall(`/api/vms/${vmId}/stop`, { method: 'POST' });
        } catch (error) {
            alert('Failed to stop VM: ' + error.message);
        }
    }
    
    async deleteVM(vmId) {
        if (confirm('Are you sure you want to delete this VM?')) {
            try {
                await this.apiCall(`/api/vms/${vmId}`, { method: 'DELETE' });
            } catch (error) {
                alert('Failed to delete VM: ' + error.message);
            }
        }
    }
    
    showLoginModal() {
        document.getElementById('login-modal').classList.remove('hidden');
    }
    
    hideLoginModal() {
        document.getElementById('login-modal').classList.add('hidden');
    }
    
    showCreateVMModal() {
        document.getElementById('create-vm-modal').classList.remove('hidden');
    }
    
    hideCreateVMModal() {
        document.getElementById('create-vm-modal').classList.add('hidden');
    }
    
    async handleCreateVM(e) {
        e.preventDefault();
        
        const vmData = {
            name: document.getElementById('vm-name').value,
            cpu_cores: parseInt(document.getElementById('vm-cpu').value),
            memory_mb: parseInt(document.getElementById('vm-memory').value),
            disk_gb: parseInt(document.getElementById('vm-disk').value),
            os_type: document.getElementById('vm-os').value
        };
        
        try {
            await this.apiCall('/api/vms', {
                method: 'POST',
                body: JSON.stringify(vmData)
            });
            
            this.hideCreateVMModal();
            this.loadDashboard();
            
            // Reset form
            document.getElementById('create-vm-form').reset();
            
        } catch (error) {
            alert('Failed to create VM: ' + error.message);
        }
    }
}

// Initialize app
const app = new NovaCronApp();