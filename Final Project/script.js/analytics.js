// analytics.js - Analytics Section Specific JavaScript

class AnalyticsManager {
    constructor() {
        this.charts = {};
        this.map = null;
        this.currentDateRange = 'Last 7 Days';
        this.init();
    }

    init() {
        this.initializeCharts();
        this.setupEventListeners();
        this.loadAnalyticsData();
    }

    initializeCharts() {
        // Incident Type Chart (Doughnut)
        this.charts.incidentType = new Chart(
            document.getElementById('incidentTypeChart'),
            {
                type: 'doughnut',
                data: {
                    labels: ['Fire', 'Falls', 'Crowd', 'Audio', 'Other'],
                    datasets: [{
                        data: [8, 5, 11, 3, 2],
                        backgroundColor: [
                            '#ff6b6b',
                            '#4ecdc4',
                            '#45b7d1',
                            '#96ceb4',
                            '#feca57'
                        ],
                        borderWidth: 2,
                        borderColor: '#fff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw || 0;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = Math.round((value / total) * 100);
                                    return `${label}: ${value} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            }
        );

        // Alerts Timeline Chart (Line)
        this.charts.alertsTimeline = new Chart(
            document.getElementById('alertsTimelineChart'),
            {
                type: 'line',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Daily Alerts',
                        data: [3, 5, 2, 6, 4, 3, 1],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#3498db',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        pointRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            }
        );

        // Detection Accuracy Chart (Bar)
        this.charts.detectionAccuracy = new Chart(
            document.getElementById('accuracyChart'),
            {
                type: 'bar',
                data: {
                    labels: ['Fire', 'Falls', 'Crowd', 'Audio'],
                    datasets: [{
                        label: 'Detection Accuracy %',
                        data: [92, 88, 85, 78],
                        backgroundColor: [
                            'rgba(76, 175, 80, 0.8)',
                            'rgba(76, 175, 80, 0.8)',
                            'rgba(255, 152, 0, 0.8)',
                            'rgba(255, 152, 0, 0.8)'
                        ],
                        borderColor: [
                            'rgba(76, 175, 80, 1)',
                            'rgba(76, 175, 80, 1)',
                            'rgba(255, 152, 0, 1)',
                            'rgba(255, 152, 0, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            }
        );

        // Response Time Chart (Bar)
        this.charts.responseTime = new Chart(
            document.getElementById('responseTimeChart'),
            {
                type: 'bar',
                data: {
                    labels: ['Fire', 'Falls', 'Crowd', 'Audio'],
                    datasets: [{
                        label: 'Avg Response (min)',
                        data: [1.8, 3.2, 2.1, 2.8],
                        backgroundColor: [
                            'rgba(76, 175, 80, 0.8)',
                            'rgba(255, 152, 0, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(255, 193, 7, 0.8)'
                        ],
                        borderColor: [
                            'rgba(76, 175, 80, 1)',
                            'rgba(255, 152, 0, 1)',
                            'rgba(255, 193, 7, 1)',
                            'rgba(255, 193, 7, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    }
                }
            }
        );
    }

    setupEventListeners() {
        // Date range selector
        const dateRangeSelect = document.getElementById('analyticsDateRange');
        if (dateRangeSelect) {
            dateRangeSelect.addEventListener('change', (e) => {
                this.currentDateRange = e.target.value;
                this.loadAnalyticsData();
            });
        }

        // Map layer selector
        const mapLayerSelect = document.getElementById('mapLayer');
        if (mapLayerSelect) {
            mapLayerSelect.addEventListener('change', (e) => {
                this.updateMapLayer(e.target.value);
            });
        }

        // Refresh map button
        const refreshMapBtn = document.getElementById('refreshMap');
        if (refreshMapBtn) {
            refreshMapBtn.addEventListener('click', () => {
                this.refreshMapData();
            });
        }

        // Load map button
        const loadMapBtn = document.getElementById('loadMapBtn');
        if (loadMapBtn) {
            loadMapBtn.addEventListener('click', () => {
                this.initializeMap();
            });
        }

        // Export report button
        const exportBtn = document.querySelector('.btn-outline');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportReport();
            });
        }
    }

    loadAnalyticsData() {
        // Simulate API call to fetch analytics data
        console.log(`Loading analytics data for: ${this.currentDateRange}`);
        
        // Show loading state
        this.showLoadingState();

        // Simulate API delay
        setTimeout(() => {
            this.updateChartsWithNewData();
            this.hideLoadingState();
        }, 1000);
    }

    updateChartsWithNewData() {
        // Generate random data based on date range for demo purposes
        const multiplier = this.getDateRangeMultiplier();
        
        // Update incident type chart
        const newIncidentData = [8, 5, 11, 3, 2].map(value => 
            Math.round(value * multiplier + Math.random() * 3 - 1.5)
        );
        this.charts.incidentType.data.datasets[0].data = newIncidentData;
        this.charts.incidentType.update();

        // Update timeline chart
        const newTimelineData = [3, 5, 2, 6, 4, 3, 1].map(value => 
            Math.round(value * multiplier + Math.random() * 2 - 1)
        );
        this.charts.alertsTimeline.data.datasets[0].data = newTimelineData;
        this.charts.alertsTimeline.update();

        // Update stats cards
        this.updateStatsCards(newIncidentData.reduce((a, b) => a + b, 0));
    }

    getDateRangeMultiplier() {
        switch(this.currentDateRange) {
            case 'Last 7 Days': return 1;
            case 'Last 30 Days': return 4;
            case 'Last 90 Days': return 12;
            case 'Year to Date': return 24;
            default: return 1;
        }
    }

    updateStatsCards(totalIncidents) {
        const statCards = document.querySelectorAll('.stat-value');
        if (statCards[0]) {
            statCards[0].textContent = totalIncidents;
            
            // Update trend indicators
            const trends = document.querySelectorAll('.stat-trend');
            trends.forEach(trend => {
                const isPositive = Math.random() > 0.5;
                const percentage = Math.floor(Math.random() * 30) + 5;
                trend.textContent = `${isPositive ? '+' : '-'}${percentage}% from last period`;
                trend.className = `stat-trend ${isPositive ? 'trend-up' : 'trend-down'}`;
            });
        }
    }

    initializeMap() {
        const mapContainer = document.getElementById('analyticsMap');
        if (!mapContainer || window.L === undefined) {
            console.warn('Leaflet not loaded or map container not found');
            return;
        }

        // Remove loading message
        mapContainer.innerHTML = '';

        // Initialize map
        this.map = L.map('analyticsMap').setView([27.7172, 85.3240], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);

        // Add sample incident markers
        this.addIncidentMarkers();
    }

    addIncidentMarkers() {
        if (!this.map) return;

        const incidents = [
            { lat: 27.7172, lng: 85.3240, type: 'fire', severity: 'high' },
            { lat: 27.7200, lng: 85.3200, type: 'crowd', severity: 'medium' },
            { lat: 27.7150, lng: 85.3280, type: 'fall', severity: 'high' },
            { lat: 27.7190, lng: 85.3220, type: 'audio', severity: 'low' },
            { lat: 27.7165, lng: 85.3260, type: 'fire', severity: 'medium' }
        ];

        incidents.forEach(incident => {
            const color = this.getMarkerColor(incident.severity);
            const icon = L.divIcon({
                className: 'custom-marker',
                html: `<div style="background-color: ${color}; width: 12px; height: 12px; border-radius: 50%; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>`,
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            });

            L.marker([incident.lat, incident.lng], { icon: icon })
                .addTo(this.map)
                .bindPopup(`
                    <strong>${incident.type.toUpperCase()} Incident</strong><br>
                    Severity: ${incident.severity}<br>
                    Location: ${incident.lat.toFixed(4)}, ${incident.lng.toFixed(4)}
                `);
        });
    }

    getMarkerColor(severity) {
        switch(severity) {
            case 'high': return '#e74c3c';
            case 'medium': return '#f39c12';
            case 'low': return '#27ae60';
            default: return '#3498db';
        }
    }

    updateMapLayer(layerType) {
        if (!this.map) return;
        
        // Clear existing layers except base tile layer
        this.map.eachLayer(layer => {
            if (!layer._url || !layer._url.includes('openstreetmap')) {
                this.map.removeLayer(layer);
            }
        });

        switch(layerType) {
            case 'heatmap':
                this.addHeatmapLayer();
                break;
            case 'markers':
                this.addIncidentMarkers();
                break;
            case 'clusters':
                this.addClusterLayer();
                break;
        }
    }

    addHeatmapLayer() {
        // Simulate heatmap data
        const heatmapData = [
            [27.7172, 85.3240, 0.8], // [lat, lng, intensity]
            [27.7200, 85.3200, 0.6],
            [27.7150, 85.3280, 0.9],
            [27.7190, 85.3220, 0.4],
            [27.7165, 85.3260, 0.7]
        ];

        if (window.L.heatLayer) {
            L.heatLayer(heatmapData, {
                radius: 25,
                blur: 15,
                maxZoom: 17,
                gradient: {0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
            }).addTo(this.map);
        }
    }

    addClusterLayer() {
        const incidents = [
            { lat: 27.7172, lng: 85.3240, type: 'fire' },
            { lat: 27.7200, lng: 85.3200, type: 'crowd' },
            { lat: 27.7150, lng: 85.3280, type: 'fall' },
            { lat: 27.7190, lng: 85.3220, type: 'audio' },
            { lat: 27.7165, lng: 85.3260, type: 'fire' }
        ];

        const markers = L.markerClusterGroup();

        incidents.forEach(incident => {
            const marker = L.marker([incident.lat, incident.lng]);
            marker.bindPopup(`<strong>${incident.type.toUpperCase()} Incident</strong>`);
            markers.addLayer(marker);
        });

        this.map.addLayer(markers);
    }

    refreshMapData() {
        if (!this.map) {
            this.initializeMap();
            return;
        }

        // Clear existing markers and re-add
        this.map.eachLayer(layer => {
            if (!layer._url || !layer._url.includes('openstreetmap')) {
                this.map.removeLayer(layer);
            }
        });

        this.addIncidentMarkers();
        
        // Show refresh feedback
        this.showToast('Map data refreshed', 'success');
    }

    exportReport() {
        // Simulate report generation and download
        this.showToast('Generating analytics report...', 'info');
        
        setTimeout(() => {
            // Create a dummy download
            const blob = new Blob(['Analytics Report Data'], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `smartzone-analytics-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showToast('Report downloaded successfully', 'success');
        }, 2000);
    }

    showLoadingState() {
        const sections = document.querySelectorAll('.chart-container, .stat-card');
        sections.forEach(section => {
            section.style.opacity = '0.7';
            section.style.pointerEvents = 'none';
        });
    }

    hideLoadingState() {
        const sections = document.querySelectorAll('.chart-container, .stat-card');
        sections.forEach(section => {
            section.style.opacity = '1';
            section.style.pointerEvents = 'auto';
        });
    }

    showToast(message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `analytics-toast analytics-toast-${type}`;
        toast.innerHTML = `
            <span>${message}</span>
            <button class="toast-close">&times;</button>
        `;

        // Add styles
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

        // Add close button event
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });

        document.body.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }

    // Cleanup method for when section is closed
    destroy() {
        // Destroy all charts
        Object.values(this.charts).forEach(chart => {
            chart.destroy();
        });
        
        // Remove map if exists
        if (this.map) {
            this.map.remove();
            this.map = null;
        }
    }
}

// Initialize analytics when section is loaded
document.addEventListener('DOMContentLoaded', function() {
    let analyticsManager = null;
    
    // Listen for section changes (this would be handled by your main dashboard)
    document.addEventListener('sectionChanged', function(e) {
        if (e.detail.section === 'analytics') {
            analyticsManager = new AnalyticsManager();
        } else if (analyticsManager) {
            analyticsManager.destroy();
            analyticsManager = null;
        }
    });

    // Manual initialization if already on analytics section
    if (document.getElementById('analytics')?.classList.contains('active')) {
        analyticsManager = new AnalyticsManager();
    }
});