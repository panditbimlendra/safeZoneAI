// Sample events data
const recentEvents = [
    {
        id: 1,
        time: "10:23 AM",
        timestamp: "2023-10-15 10:23:00",
        camera: "Parking Area",
        anomalyType: "Car Accident",
        severity: "high",
        location: "North Parking",
        status: "active"
    },
    {
        id: 2,
        time: "09:45 AM",
        timestamp: "2023-10-15 09:45:00",
        camera: "Warehouse",
        anomalyType: "Fire Detected",
        severity: "high",
        location: "Storage Area",
        status: "resolved"
    },
    {
        id: 3,
        time: "08:12 AM",
        timestamp: "2023-10-15 08:12:00",
        camera: "Common Area",
        anomalyType: "Loud Sound",
        severity: "medium",
        location: "Central Plaza",
        status: "active"
    },
    {
        id: 4,
        time: "07:30 AM",
        timestamp: "2023-10-15 07:30:00",
        camera: "Entrance Gate",
        anomalyType: "Weapon Detected",
        severity: "high",
        location: "Main Entrance",
        status: "resolved"
    },
    {
        id: 5,
        time: "06:45 AM",
        timestamp: "2023-10-15 06:45:00",
        camera: "Hallway",
        anomalyType: "Person Fall",
        severity: "medium",
        location: "First Floor",
        status: "active"
    }
];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    updateCurrentTime();
    loadRecentEvents();
    setupEventListeners();
    simulateLiveUpdates();
});

// Update current time
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('currentTime').textContent = timeString;
}

// Load recent events to table
function loadRecentEvents() {
    const eventsTable = document.getElementById('recentEvents');
    eventsTable.innerHTML = '';
    
    recentEvents.forEach(event => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>
                <strong>${event.time}</strong><br>
                <small>${event.timestamp.split(' ')[0]}</small>
            </td>
            <td>${event.camera}</td>
            <td>
                <div class="anomaly-type">
                    <i class="fas ${getAnomalyIcon(event.anomalyType)}"></i>
                    ${event.anomalyType}
                </div>
            </td>
            <td><span class="severity ${event.severity}">${event.severity.toUpperCase()}</span></td>
            <td>${event.location}</td>
            <td>
                <button class="action-btn view" onclick="viewEvent(${event.id})">
                    <i class="fas fa-eye"></i> View
                </button>
                <button class="action-btn resolve" onclick="resolveEvent(${event.id})">
                    <i class="fas fa-check"></i> Resolve
                </button>
            </td>
        `;
        eventsTable.appendChild(row);
    });
}

// Get icon for anomaly type
function getAnomalyIcon(type) {
    const icons = {
        'Car Accident': 'fa-car-crash',
        'Fire Detected': 'fa-fire',
        'Loud Sound': 'fa-volume-up',
        'Weapon Detected': 'fa-gun',
        'Person Fall': 'fa-person-falling',
        'Intrusion': 'fa-user-secret',
        'Unattended Object': 'fa-suitcase'
    };
    return icons[type] || 'fa-exclamation-triangle';
}

// Setup event listeners
function setupEventListeners() {
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        this.classList.add('rotating');
        setTimeout(() => {
            this.classList.remove('rotating');
            loadRecentEvents();
            showNotification('Dashboard refreshed successfully');
        }, 1000);
    });
    
    // Add camera button
    document.getElementById('addCameraBtn').addEventListener('click', function() {
        showNotification('Camera setup wizard will open in a new window');
    });
    
    // Navigation active state
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', function() {
            document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Camera card click
    document.querySelectorAll('.camera-card').forEach(card => {
        card.addEventListener('click', function() {
            const cameraName = this.querySelector('h4').textContent;
            showNotification(`Opening detailed view for ${cameraName}`);
        });
    });
}

// View event details
function viewEvent(eventId) {
    const event = recentEvents.find(e => e.id === eventId);
    if (event) {
        alert(`Event Details:\n\nTime: ${event.timestamp}\nCamera: ${event.camera}\nType: ${event.anomalyType}\nSeverity: ${event.severity}\nLocation: ${event.location}\nStatus: ${event.status}`);
    }
}

// Resolve event
function resolveEvent(eventId) {
    const event = recentEvents.find(e => e.id === eventId);
    if (event) {
        event.status = 'resolved';
        loadRecentEvents();
        showNotification(`Event #${eventId} marked as resolved`);
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Simulate live updates
function simulateLiveUpdates() {
    // Update time every second
    setInterval(updateCurrentTime, 1000);
    
    // Simulate new events every 30 seconds
    setInterval(() => {
        const eventTypes = ['Car Accident', 'Loud Sound', 'Person Fall', 'Intrusion'];
        const randomType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
        const randomCamera = ['Parking Area', 'Common Area', 'Entrance Gate', 'Warehouse'][Math.floor(Math.random() * 4)];
        const randomSeverity = Math.random() > 0.7 ? 'high' : Math.random() > 0.5 ? 'medium' : 'low';
        
        const newEvent = {
            id: recentEvents.length + 1,
            time: new Date().toLocaleTimeString('en-US', {hour: '2-digit', minute:'2-digit'}),
            timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
            camera: randomCamera,
            anomalyType: randomType,
            severity: randomSeverity,
            location: randomCamera === 'Parking Area' ? 'North Parking' : 
                      randomCamera === 'Warehouse' ? 'Storage Area' :
                      randomCamera === 'Common Area' ? 'Central Plaza' : 'Main Entrance',
            status: 'active'
        };
        
        recentEvents.unshift(newEvent);
        if (recentEvents.length > 10) recentEvents.pop();
        loadRecentEvents();
        
        // Show notification for high severity events
        if (randomSeverity === 'high') {
            showNotification(`New ${randomType} detected at ${randomCamera}`, 'warning');
       