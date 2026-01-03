// settings.js - Settings Section Specific JavaScript

class SettingsManager {
    constructor() {
        this.settings = this.loadSettings();
        this.unsavedChanges = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadCurrentSettings();
        this.setupFormValidation();
    }

    setupEventListeners() {
        // Master notification toggle
        const masterToggle = document.getElementById('notificationsMaster');
        if (masterToggle) {
            masterToggle.addEventListener('change', (e) => {
                this.toggleNotificationSettings(e.target.checked);
                this.markUnsavedChanges();
            });
        }

        // All toggle switches
        document.querySelectorAll('.toggle-switch input[type="checkbox"]').forEach(toggle => {
            toggle.addEventListener('change', () => {
                this.markUnsavedChanges();
            });
        });

        // All sliders
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', (e) => {
                this.updateSliderValue(e.target);
                this.markUnsavedChanges();
            });
        });

        // All form inputs
        document.querySelectorAll('input, select, textarea').forEach(input => {
            input.addEventListener('input', () => {
                this.markUnsavedChanges();
            });
        });

        // Save buttons
        document.querySelectorAll('.btn-primary').forEach(button => {
            if (button.textContent.includes('Save')) {
                button.addEventListener('click', () => {
                    this.saveSettings();
                });
            }
        });

        // Reset button
        const resetBtn = document.querySelector('.btn-outline');
        if (resetBtn && resetBtn.textContent.includes('Reset')) {
            resetBtn.addEventListener('click', () => {
                this.resetToDefaults();
            });
        }

        // Cancel button
        const cancelBtn = document.querySelector('.btn-outline');
        if (cancelBtn && cancelBtn.textContent.includes('Cancel')) {
            cancelBtn.addEventListener('click', () => {
                this.cancelChanges();
            });
        }

        // Theme selection
        document.querySelectorAll('.theme-option').forEach(theme => {
            theme.addEventListener('click', (e) => {
                this.selectTheme(e.currentTarget);
                this.markUnsavedChanges();
            });
        });

        // API key actions
        this.setupApiKeyActions();

        // Before unload warning
        window.addEventListener('beforeunload', (e) => {
            if (this.unsavedChanges) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    setupApiKeyActions() {
        // Copy API key
        const copyBtn = document.querySelector('.btn-icon .fa-copy')?.closest('.btn-icon');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                this.copyApiKey();
            });
        }

        // Regenerate API key
        const regenerateBtn = document.querySelector('.btn-icon .fa-redo')?.closest('.btn-icon');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                this.regenerateApiKey();
            });
        }
    }

    setupFormValidation() {
        // Email validation
        const emailInputs = document.querySelectorAll('input[type="email"], input[placeholder*="email"]');
        emailInputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateEmailInput(input);
            });
        });

        // Phone number validation
        const phoneInputs = document.querySelectorAll('input[placeholder*="phone"]');
        phoneInputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validatePhoneInput(input);
            });
        });

        // Number range validation
        const numberInputs = document.querySelectorAll('input[type="number"]');
        numberInputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateNumberInput(input);
            });
        });
    }

    loadCurrentSettings() {
        // Load settings from localStorage or use defaults
        Object.keys(this.settings).forEach(key => {
            const element = this.findSettingElement(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = this.settings[key];
                } else if (element.type === 'range') {
                    element.value = this.settings[key];
                    this.updateSliderValue(element);
                } else {
                    element.value = this.settings[key];
                }
            }
        });

        // Apply theme if set
        if (this.settings.theme) {
            this.applyTheme(this.settings.theme);
        }

        this.unsavedChanges = false;
        this.updateSaveButtonState();
    }

    findSettingElement(settingKey) {
        const mapping = {
            'notificationsMaster': 'notificationsMaster',
            'emailNotifications': 'emailToggle',
            'smsNotifications': 'smsToggle',
            'pushNotifications': 'pushToggle',
            'fireSensitivity': 'input[type="range"]:first-of-type',
            'fallSensitivity': 'input[type="range"]:nth-of-type(2)',
            'crowdThreshold': 'input[type="number"]:first-of-type',
            'audioSensitivity': 'input[type="range"]:nth-of-type(3)',
            'minConfidence': 'input.confidence-slider'
        };

        const selector = mapping[settingKey];
        return selector ? document.getElementById(selector) || document.querySelector(selector) : null;
    }

    toggleNotificationSettings(enabled) {
        const notificationToggles = [
            'emailToggle',
            'smsToggle',
            'pushToggle'
        ];

        notificationToggles.forEach(toggleId => {
            const toggle = document.getElementById(toggleId);
            if (toggle) {
                toggle.checked = enabled;
                toggle.disabled = !enabled;
            }
        });

        // Also disable/enable related inputs
        const notificationInputs = document.querySelectorAll('input[placeholder*="email"], input[placeholder*="phone"]');
        notificationInputs.forEach(input => {
            input.disabled = !enabled;
        });
    }

    updateSliderValue(slider) {
        const valueDisplay = slider.closest('.slider-container')?.querySelector('.slider-value');
        if (valueDisplay) {
            if (slider.classList.contains('confidence-slider')) {
                valueDisplay.textContent = `${slider.value}%`;
            } else {
                valueDisplay.textContent = `${slider.value}/10`;
            }
        }

        // Update visual indicator
        const percent = (slider.value - slider.min) / (slider.max - slider.min) * 100;
        slider.style.background = `linear-gradient(to right, #3498db ${percent}%, #ddd ${percent}%)`;
    }

    selectTheme(themeElement) {
        // Remove active class from all themes
        document.querySelectorAll('.theme-option').forEach(theme => {
            theme.classList.remove('active');
        });

        // Add active class to selected theme
        themeElement.classList.add('active');

        // Get theme name from text content
        const themeName = themeElement.querySelector('span').textContent.toLowerCase();
        this.applyTheme(themeName);
    }

    applyTheme(themeName) {
        // Remove existing theme classes
        document.body.classList.remove('theme-light', 'theme-dark', 'theme-auto');

        // Add new theme class
        document.body.classList.add(`theme-${themeName}`);

        // Store theme preference
        this.settings.theme = themeName;
    }

    copyApiKey() {
        const apiKeyInput = document.querySelector('input[value*="sk_"]');
        if (apiKeyInput) {
            navigator.clipboard.writeText(apiKeyInput.value).then(() => {
                this.showToast('API key copied to clipboard', 'success');
            }).catch(() => {
                // Fallback for older browsers
                apiKeyInput.select();
                document.execCommand('copy');
                this.showToast('API key copied to clipboard', 'success');
            });
        }
    }

    regenerateApiKey() {
        if (confirm('Are you sure you want to regenerate the API key? This will invalidate the current key.')) {
            // Show loading state
            const regenerateBtn = document.querySelector('.btn-icon .fa-redo')?.closest('.btn-icon');
            if (regenerateBtn) {
                regenerateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                regenerateBtn.disabled = true;
            }

            // Simulate API call
            setTimeout(() => {
                const newApiKey = 'sk_live_' + Math.random().toString(36).substr(2, 24) + Math.random().toString(36).substr(2, 12);
                const apiKeyInput = document.querySelector('input[value*="sk_"]');
                if (apiKeyInput) {
                    apiKeyInput.value = newApiKey;
                }

                // Reset button
                if (regenerateBtn) {
                    regenerateBtn.innerHTML = '<i class="fas fa-redo"></i>';
                    regenerateBtn.disabled = false;
                }

                this.showToast('API key regenerated successfully', 'success');
                this.markUnsavedChanges();
            }, 1500);
        }
    }

    validateEmailInput(input) {
        const emails = input.value.split(',').map(email => email.trim());
        const invalidEmails = emails.filter(email => !this.isValidEmail(email) && email !== '');

        if (invalidEmails.length > 0) {
            this.showInputError(input, 'Contains invalid email addresses');
            return false;
        } else {
            this.clearInputError(input);
            return true;
        }
    }

    validatePhoneInput(input) {
        const phones = input.value.split(',').map(phone => phone.trim());
        const invalidPhones = phones.filter(phone => !this.isValidPhone(phone) && phone !== '');

        if (invalidPhones.length > 0) {
            this.showInputError(input, 'Contains invalid phone numbers');
            return false;
        } else {
            this.clearInputError(input);
            return true;
        }
    }

    validateNumberInput(input) {
        const value = parseFloat(input.value);
        const min = parseFloat(input.min) || -Infinity;
        const max = parseFloat(input.max) || Infinity;

        if (isNaN(value) || value < min || value > max) {
            this.showInputError(input, `Must be between ${min} and ${max}`);
            return false;
        } else {
            this.clearInputError(input);
            return true;
        }
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isValidPhone(phone) {
        const phoneRegex = /^\+?[\d\s\-\(\)]{10,}$/;
        return phoneRegex.test(phone);
    }

    showInputError(input, message) {
        this.clearInputError(input);
        
        input.style.borderColor = '#e74c3c';
        
        const errorElement = document.createElement('div');
        errorElement.className = 'input-error';
        errorElement.textContent = message;
        errorElement.style.cssText = `
            color: #e74c3c;
            font-size: 12px;
            margin-top: 4px;
        `;
        
        input.parentNode.appendChild(errorElement);
    }

    clearInputError(input) {
        input.style.borderColor = '';
        const existingError = input.parentNode.querySelector('.input-error');
        if (existingError) {
            existingError.remove();
        }
    }

    markUnsavedChanges() {
        this.unsavedChanges = true;
        this.updateSaveButtonState();
    }

    updateSaveButtonState() {
        const saveButtons = document.querySelectorAll('.btn-primary');
        saveButtons.forEach(button => {
            if (button.textContent.includes('Save')) {
                if (this.unsavedChanges) {
                    button.disabled = false;
                    button.style.opacity = '1';
                } else {
                    button.disabled = true;
                    button.style.opacity = '0.6';
                }
            }
        });
    }

    saveSettings() {
        // Validate all inputs before saving
        if (!this.validateAllInputs()) {
            this.showToast('Please fix validation errors before saving', 'error');
            return;
        }

        // Show saving state
        this.showToast('Saving settings...', 'info');

        // Collect all settings
        this.collectCurrentSettings();

        // Simulate API call
        setTimeout(() => {
            this.saveToStorage();
            this.unsavedChanges = false;
            this.updateSaveButtonState();
            this.showToast('Settings saved successfully', 'success');
        }, 1000);
    }

    validateAllInputs() {
        let isValid = true;

        // Validate emails
        const emailInputs = document.querySelectorAll('input[placeholder*="email"]');
        emailInputs.forEach(input => {
            if (!this.validateEmailInput(input)) {
                isValid = false;
            }
        });

        // Validate phones
        const phoneInputs = document.querySelectorAll('input[placeholder*="phone"]');
        phoneInputs.forEach(input => {
            if (!this.validatePhoneInput(input)) {
                isValid = false;
            }
        });

        // Validate numbers
        const numberInputs = document.querySelectorAll('input[type="number"]');
        numberInputs.forEach(input => {
            if (!this.validateNumberInput(input)) {
                isValid = false;
            }
        });

        return isValid;
    }

    collectCurrentSettings() {
        // Collect toggle states
        document.querySelectorAll('.toggle-switch input[type="checkbox"]').forEach(toggle => {
            const key = toggle.id.replace('Toggle', '');
            this.settings[key] = toggle.checked;
        });

        // Collect slider values
        document.querySelectorAll('input[type="range"]').forEach(slider => {
            const key = slider.classList.contains('confidence-slider') ? 'minConfidence' : 'sensitivity';
            this.settings[key] = slider.value;
        });

        // Collect input values
        document.querySelectorAll('input:not([type="checkbox"]):not([type="range"]), select, textarea').forEach(input => {
            const key = this.getSettingKeyFromInput(input);
            if (key) {
                this.settings[key] = input.value;
            }
        });

        // Collect theme
        const activeTheme = document.querySelector('.theme-option.active');
        if (activeTheme) {
            this.settings.theme = activeTheme.querySelector('span').textContent.toLowerCase();
        }
    }

    getSettingKeyFromInput(input) {
        const placeholder = input.placeholder?.toLowerCase() || '';
        const name = input.name?.toLowerCase() || '';
        
        if (placeholder.includes('email') || name.includes('email')) return 'emailAddresses';
        if (placeholder.includes('phone') || name.includes('phone')) return 'phoneNumbers';
        if (placeholder.includes('api')) return 'apiEndpoint';
        if (placeholder.includes('url')) return 'webhookUrl';
        
        return null;
    }

    saveToStorage() {
        try {
            localStorage.setItem('smartzoneSettings', JSON.stringify(this.settings));
        } catch (e) {
            console.warn('Could not save settings to localStorage:', e);
        }
    }

    loadSettings() {
        try {
            const saved = localStorage.getItem('smartzoneSettings');
            return saved ? JSON.parse(saved) : this.getDefaultSettings();
        } catch (e) {
            console.warn('Could not load settings from localStorage:', e);
            return this.getDefaultSettings();
        }
    }

    getDefaultSettings() {
        return {
            notificationsMaster: true,
            emailNotifications: true,
            smsNotifications: true,
            pushNotifications: true,
            emailAddresses: 'admin@company.com, security@company.com',
            phoneNumbers: '+1234567890, +0987654321',
            fireSensitivity: 7,
            fallSensitivity: 8,
            crowdThreshold: 10,
            audioSensitivity: 6,
            minConfidence: 75,
            theme: 'light'
        };
    }

    resetToDefaults() {
        if (confirm('Are you sure you want to reset all settings to default values?')) {
            this.settings = this.getDefaultSettings();
            this.loadCurrentSettings();
            this.showToast('Settings reset to defaults', 'success');
        }
    }

    cancelChanges() {
        if (this.unsavedChanges) {
            if (confirm('You have unsaved changes. Are you sure you want to cancel?')) {
                this.loadCurrentSettings();
                this.showToast('Changes discarded', 'info');
            }
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `settings-toast settings-toast-${type}`;
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

// Initialize settings manager when section is loaded
document.addEventListener('DOMContentLoaded', function() {
    let settingsManager = null;
    
    document.addEventListener('sectionChanged', function(e) {
        if (e.detail.section === 'settings') {
            settingsManager = new SettingsManager();
        }
    });

    if (document.getElementById('settings')?.classList.contains('active')) {
        settingsManager = new SettingsManager();
    }
});