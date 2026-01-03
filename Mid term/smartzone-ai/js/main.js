// Main application controller
class DashboardApp {
    constructor() {
        this.initializeApp();
    }

    initializeApp() {
        this.setupSidebar();
        this.setupTimeDisplay();
    }

    setupSidebar() {
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');

        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
            });

            if (localStorage.getItem('sidebarCollapsed') === 'true') {
                sidebar.classList.add('collapsed');
            }
        }
    }

    setupTimeDisplay() {
        this.updateKathmanduTime();
    }

    updateKathmanduTime() {
        const now = new Date();
        const offset = 5.75 * 60 * 60 * 1000; // UTC+5:45
        const kathmanduTime = new Date(now.getTime() + offset);
        
        const dateStr = kathmanduTime.toISOString().split('T')[0];
        const timeStr = kathmanduTime.toTimeString().substring(0, 8);
        
        setTimeout(() => this.updateKathmanduTime(), 1000);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new DashboardApp();
});