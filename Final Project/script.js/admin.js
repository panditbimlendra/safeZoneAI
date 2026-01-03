// admin.js - Admin Section Specific JavaScript

class AdminManager {
    constructor() {
        this.users = this.loadUsers();
        this.logs = this.loadLogs();
        this.currentLogPage = 1;
        this.logsPerPage = 10;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.renderUsers();
        this.renderLogs();
        this.loadSystemMetrics();
        this.setupRealTimeUpdates();
    }

    setupEventListeners() {
        // User management
        this.setupUserManagement();

        // Log controls
        this.setupLogControls();

        // System maintenance actions
        this.setupMaintenanceActions();

        // Refresh system button
        const refreshBtn = document.querySelector('.btn[title*="Refresh"]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshSystem();
            });
        }

        // Export logs button
        const exportBtn = document.querySelector('.btn[title*="Export"]');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportLogs();
            });
        }
    }

    setupUserManagement() {
        // Add user form
        const addUserForm = document.querySelector('.input-with-button');
        if (addUserForm) {
            const input = addUserForm.querySelector('input');
            const button = addUserForm.querySelector('button');
            
            button.addEventListener('click', () => {
                this.addUser(input.value);
            });

            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.addUser(input.value);
                }
            });
        }

        // User role selection
        const roleSelect = document.querySelector('select');
        if (roleSelect) {
            roleSelect.addEventListener('change', () => {
                this.updateNewUserRole(roleSelect.value);
            });
        }

        // Permission checkboxes
        document.querySelectorAll('.checkbox-item input').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateNewUserPermissions();
            });
        });
    }

    setupLogControls() {
        // Log filter
        const logFilter = document.querySelector('.log-controls select');
        if (logFilter) {
            logFilter.addEventListener('change', (e) => {
                this.filterLogs(e.target.value);
            });
        }

        // Log search
        const searchBtn = document.querySelector('.log-controls .fa-search')?.closest('.btn-icon');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                this.searchLogs();
            });
        }

        // Log pagination
        const prevBtn = document.querySelector('.log-pagination .fa-chevron-left')?.closest('.btn-icon');
        const nextBtn = document.querySelector('.log-pagination .fa-chevron-right')?.closest('.btn-icon');
        
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                this.previousLogPage();
            });
        }

        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                this.nextLogPage();
            });
        }
    }

    setupMaintenanceActions() {
        const actionButtons = document.querySelectorAll('.btn-action');
        actionButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.currentTarget.querySelector('span').textContent;
                this.executeMaintenanceAction(action);
            });
        });

        // Backup schedule
        const backupSchedule = document.querySelector('select');
        if (backupSchedule && backupSchedule.parentElement.querySelector('label')?.textContent.includes('Backup')) {
            backupSchedule.addEventListener('change', () => {
                this.updateBackupSchedule(backupSchedule.value);
            });
        }

        // Update download
        const updateBtn = document.querySelector('.btn-primary');
        if (updateBtn && updateBtn.textContent.includes('Download')) {
            updateBtn.addEventListener('click', () => {
                this.downloadUpdate();
            });
        }
    }

    addUser(email) {
        if (!this.isValidEmail(email)) {
            this.showToast('Please enter a valid email address', 'error');
            return;
        }

        if (this.users.find(user => user.email === email)) {
            this.showToast('User already exists', 'error');
            return;
        }

        const roleSelect = document.querySelector('select');
        const role = roleSelect ? roleSelect.value : 'Viewer';

        const permissions = this.getSelectedPermissions();

        const newUser = {
            id: this.generateId(),
            email: email,
            role: role,
            permissions: permissions,
            avatar: this.getAvatarForRole(role),
            createdAt: new Date().toISOString()
        };

        this.users.push(newUser);
        this.saveUsers();
        this.renderUsers();
        
        // Clear input
        const input = document.querySelector('.input-with-button input');
        if (input) input.value = '';

        this.showToast(`User ${email} added successfully`, 'success');
    }

    updateNewUserRole(role) {
        // Update permissions based on role
        const permissions = this.getDefaultPermissionsForRole(role);
        this.setPermissionsSelection(permissions);
    }

    updateNewUserPermissions() {
        // Permissions are updated in real-time through checkbox changes
    }

    getSelectedPermissions() {
        const permissions = [];
        document.querySelectorAll('.checkbox-item input:checked').forEach(checkbox => {
            permissions.push(checkbox.parentElement.textContent.trim());
        });
        return permissions;
    }

    setPermissionsSelection(permissions) {
        document.querySelectorAll('.checkbox-item input').forEach(checkbox => {
            const permission = checkbox.parentElement.textContent.trim();
            checkbox.checked = permissions.includes(permission);
        });
    }

    getDefaultPermissionsForRole(role) {
        const rolePermissions = {
            'Administrator': ['View Live Feeds', 'Manage Alerts', 'System Configuration', 'User Management', 'Data Export'],
            'Operator': ['View Live Feeds', 'Manage Alerts', 'Data Export'],
            'Viewer': ['View Live Feeds'],
            'Auditor': ['View Live Feeds', 'Data Export']
        };
        
        return rolePermissions[role] || ['View Live Feeds'];
    }

    editUser(userId) {
        const user = this.users.find(u => u.id === userId);
        if (user) {
            // In a real app, this would open a modal
            this.showToast(`Editing user: ${user.email}`, 'info');
        }
    }

    removeUser(userId) {
        const user = this.users.find(u => u.id === userId);
        if (user && confirm(`Are you sure you want to remove user ${user.email}?`)) {
            this.users = this.users.filter(u => u.id !== userId);
            this.saveUsers();
            this.renderUsers();
            this.showToast(`User ${user.email} removed`, 'success');
        }
    }

    renderUsers() {
        const usersList = document.querySelector('.users-list');
        if (!usersList) return;

        const usersHTML = this.users.map(user => `
            <div class="user-item">
                <div class="user-avatar">
                    <i class="${user.avatar}"></i>
                </div>
                <div class="user-info">
                    <strong>${user.email}</strong>
                    <span class="user-role">${user.role}</span>
                </div>
                <div class="user-actions">
                    <button class="btn-icon" onclick="adminManager.editUser('${user.id}')" title="Edit User">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn-icon btn-danger" onclick="adminManager.removeUser('${user.id}')" title="Remove User">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');

        usersList.querySelector('.user-item')?.remove(); // Remove template if exists
        usersList.insertAdjacentHTML('beforeend', usersHTML);
    }

    filterLogs(filter) {
        let filteredLogs = this.logs;

        switch(filter) {
            case 'Errors Only':
                filteredLogs = this.logs.filter(log => log.level === 'ERROR');
                break;
            case 'Warnings Only':
                filteredLogs = this.logs.filter(log => log.level === 'WARNING');
                break;
            case 'Info Only':
                filteredLogs = this.logs.filter(log => log.level === 'INFO');
                break;
        }

        this.renderLogs(filteredLogs);
    }

    searchLogs() {
        const searchTerm = prompt('Enter search term:');
        if (searchTerm) {
            const filteredLogs = this.logs.filter(log => 
                log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
                log.level.toLowerCase().includes(searchTerm.toLowerCase())
            );
            this.renderLogs(filteredLogs);
        }
    }

    previousLogPage() {
        if (this.currentLogPage > 1) {
            this.currentLogPage--;
            this.renderLogs();
        }
    }

    nextLogPage() {
        const totalPages = Math.ceil(this.logs.length / this.logsPerPage);
        if (this.currentLogPage < totalPages) {
            this.currentLogPage++;
            this.renderLogs();
        }
    }

    renderLogs(logs = this.logs) {
        const logContainer = document.querySelector('.log-container');
        const paginationInfo = document.querySelector('.page-info');
        
        if (!logContainer) return;

        // Calculate pagination
        const startIndex = (this.currentLogPage - 1) * this.logsPerPage;
        const endIndex = startIndex + this.logsPerPage;
        const paginatedLogs = logs.slice(startIndex, endIndex);

        const logsHTML = paginatedLogs.map(log => `
            <div class="log-entry log-${log.level.toLowerCase()}">
                <span class="log-time">[${log.timestamp}]</span>
                <span class="log-level">${log.level}</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');

        logContainer.innerHTML = logsHTML;

        // Update pagination info
        if (paginationInfo) {
            const totalPages = Math.ceil(logs.length / this.logsPerPage);
            paginationInfo.textContent = `Page ${this.currentLogPage} of ${totalPages}`;
        }
    }

    executeMaintenanceAction(action) {
        const actions = {
            'Run Backup': () => this.runBackup(),
            'Check Updates': () => this.checkUpdates(),
            'Clear Cache': () => this.clearCache(),
            'Restart Services': () => this.restartServices(),
            'Restart System': () => this.restartSystem(),
            'Emergency Stop': () => this.emergencyStop()
        };

        if (actions[action]) {
            actions[action]();
        }
    }

    runBackup() {
        this.showToast('Starting system backup...', 'info');
        
        // Simulate backup process
        setTimeout(() => {
            this.showToast('System backup completed successfully', 'success');
            this.addLog('INFO', 'System backup completed');
        }, 3000);
    }

    checkUpdates() {
        this.showToast('Checking for updates...', 'info');
        
        setTimeout(() => {
            // Simulate update check result
            const hasUpdates = Math.random() > 0.5;
            if (hasUpdates) {
                this.showToast('New update available: v2.2.0', 'warning');
            } else {
                this.showToast('System is up to date', 'success');
            }
        }, 2000);
    }

    clearCache() {
        if (confirm('Are you sure you want to clear all cache?')) {
            this.showToast('Clearing cache...', 'info');
            
            setTimeout(() => {
                this.showToast('Cache cleared successfully', 'success');
                this.addLog('INFO', 'System cache cleared');
            }, 1500);
        }
    }

    restartServices() {
        if (confirm('Are you sure you want to restart all services? This may cause temporary downtime.')) {
            this.showToast('Restarting services...', 'warning');
            
            setTimeout(() => {
                this.showToast('Services restarted successfully', 'success');
                this.addLog('INFO', 'All services restarted');
            }, 4000);
        }
    }

    restartSystem() {
        if (confirm('WARNING: This will restart the entire system. Are you sure?')) {
            this.showToast('System restart initiated...', 'warning');
            
            setTimeout(() => {
                this.showToast('System restart completed', 'success');
                this.addLog('INFO', 'System restart completed');
                this.loadSystemMetrics(); // Refresh metrics after restart
            }, 5000);
        }
    }

    emergencyStop() {
        if (confirm('EMERGENCY STOP: This will immediately halt all monitoring and AI processing. Continue?')) {
            this.showToast('EMERGENCY STOP ACTIVATED', 'error');
            
            setTimeout(() => {
                this.showToast('All systems halted. Manual restart required.', 'error');
                this.addLog('ERROR', 'EMERGENCY STOP: All systems halted');
            }, 2000);
        }
    }

    refreshSystem() {
        this.showToast('Refreshing system data...', 'info');
        
        // Refresh all data
        this.loadSystemMetrics();
        this.renderLogs();
        this.renderUsers();
        
        setTimeout(() => {
            this.showToast('System data refreshed', 'success');
        }, 1000);
    }

    exportLogs() {
        this.showToast('Exporting system logs...', 'info');
        
        setTimeout(() => {
            const logData = this.logs.map(log => 
                `[${log.timestamp}] ${log.level}: ${log.message}`
            ).join('\n');
            
            const blob = new Blob([logData], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `smartzone-logs-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showToast('Logs exported successfully', 'success');
        }, 2000);
    }

    downloadUpdate() {
        this.showToast('Downloading update v2.2.0...', 'info');
        
        // Simulate download progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += 10;
            this.showToast(`Downloading update: ${progress}%`, 'info');
            
            if (progress >= 100) {
                clearInterval(interval);
                setTimeout(() => {
                    this.showToast('Update downloaded. Ready to install.', 'success');
                }, 500);
            }
        }, 500);
    }

    updateBackupSchedule(schedule) {
        this.showToast(`Backup schedule updated to: ${schedule}`, 'success');
        this.addLog('INFO', `Backup schedule changed to: ${schedule}`);
    }

    loadSystemMetrics() {
        // Simulate loading system metrics
        const metrics = {
            cpu: Math.floor(Math.random() * 100),
            memory: Math.floor(Math.random() * 100),
            storage: Math.floor(Math.random() * 100),
            database: Math.floor(Math.random() * 100)
        };

        // Update metric bars
        Object.keys(metrics).forEach(metric => {
            const bar = document.querySelector(`.${metric}-usage`);
            if (bar) {
                bar.style.width = `${metrics[metric]}%`;
            }
        });

        // Update status indicators
        this.updateStatusIndicators(metrics);
    }

    updateStatusIndicators(metrics) {
        // Update CPU status
        const cpuStatus = document.querySelector('.metric-item:nth-child(3) .metric-value');
        if (cpuStatus) {
            cpuStatus.textContent = metrics.cpu > 80 ? 'High' : metrics.cpu > 60 ? 'Medium' : 'Normal';
            cpuStatus.style.background = metrics.cpu > 80 ? '#ffebee' : metrics.cpu > 60 ? '#fff3e0' : '#e8f5e8';
            cpuStatus.style.color = metrics.cpu > 80 ? '#e74c3c' : metrics.cpu > 60 ? '#f39c12' : '#27ae60';
        }
    }

    setupRealTimeUpdates() {
        // Simulate real-time log updates
        setInterval(() => {
            if (Math.random() > 0.7) { // 30% chance to add a new log
                const levels = ['INFO', 'WARNING', 'ERROR'];
                const messages = [
                    'Camera feed quality check completed',
                    'Network latency detected',
                    'User session started',
                    'Database optimization running',
                    'Security scan in progress'
                ];
                
                const level = levels[Math.floor(Math.random() * levels.length)];
                const message = messages[Math.floor(Math.random() * messages.length)];
                
                this.addLog(level, message);
            }
        }, 10000); // Every 10 seconds
    }

    addLog(level, message) {
        const newLog = {
            timestamp: new Date().toISOString().replace('T', ' ').substr(0, 19),
            level: level,
            message: message
        };

        this.logs.unshift(newLog); // Add to beginning
        this.saveLogs();
        
        // Only re-render if we're on the first page
        if (this.currentLogPage === 1) {
            this.renderLogs();
        }
    }

    loadUsers() {
        try {
            const saved = localStorage.getItem('smartzoneUsers');
            return saved ? JSON.parse(saved) : this.getDefaultUsers();
        } catch (e) {
            console.warn('Could not load users from localStorage:', e);
            return this.getDefaultUsers();
        }
    }

    getDefaultUsers() {
        return [
            {
                id: '1',
                email: 'admin@smartzone.ai',
                role: 'Administrator',
                permissions: ['View Live Feeds', 'Manage Alerts', 'System Configuration', 'User Management', 'Data Export'],
                avatar: 'fas fa-user-shield',
                createdAt: '2025-01-01T00:00:00Z'
            },
            {
                id: '2',
                email: 'operator@smartzone.ai',
                role: 'Operator',
                permissions: ['View Live Feeds', 'Manage Alerts', 'Data Export'],
                avatar: 'fas fa-user-tie',
                createdAt: '2025-01-02T00:00:00Z'
            },
            {
                id: '3',
                email: 'viewer@smartzone.ai',
                role: 'Viewer',
                permissions: ['View Live Feeds'],
                avatar: 'fas fa-user',
                createdAt: '2025-01-03T00:00:00Z'
            }
        ];
    }

    loadLogs() {
        try {
            const saved = localStorage.getItem('smartzoneLogs');
            return saved ? JSON.parse(saved) : this.getDefaultLogs();
        } catch (e) {
            console.warn('Could not load logs from localStorage:', e);
            return this.getDefaultLogs();
        }
    }

    getDefaultLogs() {
        return [
            { timestamp: '2025-08-06 10:25:32', level: 'INFO', message: 'System started successfully' },
            { timestamp: '2025-08-06 10:24:15', level: 'WARNING', message: 'High CPU usage detected (92%)' },
            { timestamp: '2025-08-06 10:24:01', level: 'ALERT', message: 'Crowd formation detected at Main Entrance' },
            { timestamp: '2025-08-06 10:22:45', level: 'INFO', message: 'Camera feed restored: Parking Lot' },
            { timestamp: '2025-08-06 10:20:30', level: 'INFO', message: 'Security update applied' },
            { timestamp: '2025-08-06 09:45:22', level: 'ALERT', message: 'Fire detected in Kitchen, Building B' },
            { timestamp: '2025-08-06 09:40:10', level: 'INFO', message: 'Database backup completed' },
            { timestamp: '2025-08-06 09:30:05', level: 'INFO', message: 'User login: admin@smartzone.ai' },
            { timestamp: '2025-08-06 08:45:18', level: 'ERROR', message: 'Camera connection lost: Staircase 3' },
            { timestamp: '2025-08-06 08:30:45', level: 'ALERT', message: 'Person fall detected at Staircase, Floor 3' }
        ];
    }

    saveUsers() {
        try {
            localStorage.setItem('smartzoneUsers', JSON.stringify(this.users));
        } catch (e) {
            console.warn('Could not save users to localStorage:', e);
        }
    }

    saveLogs() {
        try {
            // Keep only last 1000 logs to prevent localStorage overflow
            const logsToSave = this.logs.slice(0, 1000);
            localStorage.setItem('smartzoneLogs', JSON.stringify(logsToSave));
        } catch (e) {
            console.warn('Could not save logs to localStorage:', e);
        }
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    getAvatarForRole(role) {
        const avatars = {
            'Administrator': 'fas fa-user-shield',
            'Operator': 'fas fa-user-tie',
            'Viewer': 'fas fa-user',
            'Auditor': 'fas fa-chart-bar'
        };
        return avatars[role] || 'fas fa-user';
    }

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `admin-toast admin-toast-${type}`;
        toast.innerHTML = `
            <span>${message}</span>
            <button class="toast-close">&times;</button>
        `;

        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#27ae60' : type === 'error' ? '#e74c3c' : '#3498db'};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
            animation: slideInRight 0.3s ease;
        `;

        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });

        document.body.appendChild(toast);

        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize admin manager when section is loaded
let adminManager = null;

document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('sectionChanged', function(e) {
        if (e.detail.section === 'admin') {
            adminManager = new AdminManager();
        }
    });

    if (document.getElementById('admin')?.classList.contains('active')) {
        adminManager = new AdminManager();
    }
});