// Settings page specific functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeSettings();
    setupSettingsEventListeners();
});

function initializeSettings() {
    // Set up tab navigation
    setupSettingsTabs();
    
    // Load saved settings
    loadSavedSettings();
    
    // Initialize camera toggles
    initializeCameraToggles();
}

function setupSettingsTabs() {
    const menuItems = document.querySelectorAll('.settings-menu-item');
    const sections = document.querySelectorAll('.settings-section');
    
    menuItems.forEach(item => {
        item.addEventListener('click', function() {
            const sectionId = this.getAttribute('data-section');
            
            // Update active menu item
            menuItems.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding section
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === `${sectionId}-section`) {
                    section.classList.add('active');
                }
            });
            
            // Update URL hash
            window.location.hash = `settings-${sectionId}`;
        });
    });
    
    // Check URL hash on load
    if (window.location.hash) {
        const hash = window.location.hash.replace('#settings-', '');
        const correspondingItem = document.querySelector(`.settings-menu-item[data-section="${hash}"]`);
        if (correspondingItem) {
            correspondingItem.click();
        }
    }
}

function loadSavedSettings() {
    // In a real app, this would load from localStorage or API
    const savedSettings = {
        systemName: 'SmartVision AI Surveillance',
        timezone: 'UTC-5',
        dateFormat: 'MM/DD/YYYY',
        language: 'en',
        darkMode: true,
        animations: true,
        notifications: true,
        detectionConfidence: 85,
        frameRate: 10,
        aiModel: 'accurate',
        alertCooldown: 30,
        notifyEmail: true,
        notifySMS: true,
        notifyPush: true,
        notifySound: true
    };
    
    // Apply saved settings to form elements
    Object.keys(savedSettings).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = savedSettings[key];
            } else if (element.type === 'range') {
                element.value = savedSettings[key];
                // Update any associated display
                const display = document.getElementById(`${key}Display`);
                if (display) {
                    display.textContent = `${savedSettings[key]}%`;
                }
            } else {
                element.value = savedSettings[key];
            }
        }
    });
}

function initializeCameraToggles() {
    document.querySelectorAll('.camera-toggle input').forEach(toggle => {
        toggle.addEventListener('change', function() {
            const card = this.closest('.camera-setting-card');
            if (this.checked) {
                card.classList.add('active');
                showNotification('Camera activated');
            } else {
                card.classList.remove('active');
                showNotification('Camera deactivated');
            }
        });
    });
}

function setupSettingsEventListeners() {
    // Save buttons
    document.querySelectorAll('.btn-primary').forEach(btn => {
        if (btn.textContent.includes('Save')) {
            btn.addEventListener('click', function() {
                saveSettings();
            });
        }
    });
    
    // Reset buttons
    document.querySelectorAll('.btn-outline').forEach(btn => {
        if (btn.textContent.includes('Reset') || btn.textContent.includes('Default')) {
            btn.addEventListener('click', function() {
                if (confirm('Reset settings to default values?')) {
                    resetSettings();
                }
            });
        }
    });
    
    // Add camera button
    const addCameraBtn = document.getElementById('addCameraBtn');
    if (addCameraBtn) {
        addCameraBtn.addEventListener('click', function() {
            addNewCamera();
        });
    }
    
    // Add rule button
    const addRuleBtn = document.getElementById('addRuleBtn');
    if (addRuleBtn) {
        addRuleBtn.addEventListener('click', function() {
            addNewRule();
        });
    }
    
    // Test alert system button
    const testAlertBtn = document.querySelector('button:contains("Test Alert System")');
    if (testAlertBtn) {
        testAlertBtn.addEventListener('click', function() {
            testAlertSystem();
        });
    }
    
    // Train AI model button
    const trainModelBtn = document.querySelector('button:contains("Train AI Model")');
    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', function() {
            trainAIModel();
        });
    }
    
    // Range inputs with display
    document.querySelectorAll('input[type="range"]').forEach(range => {
        range.addEventListener('input', function() {
            const displayId = `${this.id}Display`;
            const display = document.getElementById(displayId);
            if (display) {
                display.textContent = `${this.value}%`;
            }
        });
    });
    
    // Camera name inputs
    document.querySelectorAll('.camera-name-input').forEach(input => {
        input.addEventListener('change', function() {
            showNotification('Camera name updated');
        });
    });
}

function saveSettings() {
    const settings = {};
    
    // Collect all form values
    document.querySelectorAll('input, select, textarea').forEach(element => {
        if (element.id) {
            if (element.type === 'checkbox') {
                settings[element.id] = element.checked;
            } else if (element.type === 'radio') {
                if (element.checked) {
                    settings[element.name] = element.value;
                }
            } else {
                settings[element.id] = element.value;
            }
        }
    });
    
    // In a real app, save to localStorage or send to API
    localStorage.setItem('smartvision-settings', JSON.stringify(settings));
    
    showNotification('Settings saved successfully', 'success');
    
    // Update system status in footer
    updateSystemStatus();
}

function resetSettings() {
    if (confirm('Are you sure you want to reset all settings to default values?')) {
        // Clear localStorage
        localStorage.removeItem('smartvision-settings');
        
        // Reload page to apply defaults
        location.reload();
    }
}

function addNewCamera() {
    const cameraGrid = document.querySelector('.camera-grid-settings');
    if (!cameraGrid) return;
    
    const cameraCount = cameraGrid.children.length + 1;
    const newCamera = document.createElement('div');
    newCamera.className = 'camera-setting-card active';
    newCamera.innerHTML = `
        <div class="camera-header-setting">
            <input type="text" class="camera-name-input" value="New Camera ${cameraCount}">
            <label class="camera-toggle">
                <input type="checkbox" checked>
                <span class="camera-toggle-slider"></span>
            </label>
        </div>
        <div class="form-group">
            <label>Camera URL</label>
            <input type="text" class="form-control" placeholder="rtsp://camera-ip:554/stream">
        </div>
        <div class="form-group">
            <label>Detection Sensitivity</label>
            <input type="range" min="1" max="10" value="5" class="form-control">
        </div>
    `;
    
    cameraGrid.appendChild(newCamera);
    initializeCameraToggles();
    
    showNotification('New camera added. Please configure the camera URL.');
}

function addNewRule() {
    const rulesContainer = document.querySelector('.alert-rules');
    if (!rulesContainer) return;
    
    const ruleTypes = [
        { type: 'accident', icon: 'fa-car-crash', name: 'Accident Detection' },
        { type: 'fall', icon: 'fa-person-falling', name: 'Person Fall' },
        { type: 'sound', icon: 'fa-volume-up', name: 'Loud Sound' },
        { type: 'intrusion', icon: 'fa-user-secret', name: 'Intrusion' }
    ];
    
    const randomType = ruleTypes[Math.floor(Math.random() * ruleTypes.length)];
    
    const newRule = document.createElement('div');
    newRule.className = 'rule-item';
    newRule.innerHTML = `
        <div class="rule-info">
            <div class="rule-icon ${randomType.type}">
                <i class="fas ${randomType.icon}"></i>
            </div>
            <div>
                <h4>${randomType.name}</h4>
                <p>New alert rule - configure actions</p>
            </div>
        </div>
        <div class="rule-actions">
            <button class="btn-sm btn-outline" onclick="editRule(this)">
                <i class="fas fa-edit"></i> Edit
            </button>
            <button class="btn-sm btn-outline" onclick="deleteRule(this)">
                <i class="fas fa-trash"></i> Delete
            </button>
        </div>
    `;
    
    rulesContainer.appendChild(newRule);
    showNotification('New alert rule added');
}

function editRule(button) {
    const ruleItem = button.closest('.rule-item');
    const ruleName = ruleItem.querySelector('h4').textContent;
    showNotification(`Editing rule: ${ruleName}`);
}

function deleteRule(button) {
    const ruleItem = button.closest('.rule-item');
    const ruleName = ruleItem.querySelector('h4').textContent;
    
    if (confirm(`Delete rule: ${ruleName}?`)) {
        ruleItem.remove();
        showNotification('Rule deleted', 'success');
    }
}

function testAlertSystem() {
    showNotification('Testing alert system...', 'info');
    
    // Simulate different alert types
    const alertTypes = ['accident', 'fire', 'weapon', 'fall'];
    const randomType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
    
    setTimeout(() => {
        const alertNames = {
            accident: 'Car Accident',
            fire: 'Fire',
            weapon: 'Weapon',
            fall: 'Person Fall'
        };
        
        showNotification(`Test alert: ${alertNames[randomType]} detected`, 'warning');
        
        // Show visual indicator
        const testAlert = document.createElement('div');
        testAlert.className = 'test-alert';
        testAlert.innerHTML = `
            <div class="alert-icon ${randomType}">
                <i class="fas ${getAlertIcon(randomType)}"></i>
            </div>
            <div>
                <strong>Test Alert</strong>
                <p>${alertNames[randomType]} detection test successful</p>
            </div>
            <button onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add styles for test alert
        const style = document.createElement('style');
        style.textContent = `
            .test-alert {
                position: fixed;
                top: 100px;
                right: 20px;
                background: white;
                border-left: 4px solid var(--warning-color);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                display: flex;
                align-items: center;
                gap: 15px;
                z-index: 9999;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); }
                to { transform: translateX(0); }
            }
            
            .test-alert button {
                background: none;
                border: none;
                color: var(--gray-dark);
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(testAlert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (testAlert.parentElement) {
                testAlert.remove();
            }
        }, 5000);
    }, 1000);
}

function trainAIModel() {
    showNotification('Starting AI model training...', 'info');
    
    // Simulate training process
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        
        if (progress <= 100) {
            showNotification(`Training AI model: ${progress}% complete`, 'info');
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            setTimeout(() => {
                showNotification('AI model training completed successfully!', 'success');
            }, 500);
        }
    }, 300);
}

function getAlertIcon(type) {
    const icons = {
        'accident': 'fa-car-crash',
        'fire': 'fa-fire',
        'weapon': 'fa-gun',
        'fall': 'fa-person-falling'
    };
    return icons[type] || 'fa-exclamation-triangle';
}

function updateSystemStatus() {
    const statusElement = document.querySelector('.system-status');
    if (statusElement) {
        const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        statusElement.textContent = `Settings Updated: ${time}`;
    }
}