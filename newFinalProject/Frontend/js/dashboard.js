

// Dashboard specific functionality
document.addEventListener("DOMContentLoaded", function () {
  initializeDashboard();
  setupDashboardEventListeners();
});

function initializeDashboard() {
  // Initialize dashboard components
  updateDashboardStats();
  loadCameraFeeds();
  setupRealTimeUpdates();
}

function updateDashboardStats() {
  // Update dashboard statistics
  const stats = {
    accidents: 12,
    fire: 3,
    weapons: 2,
    falls: 8,
    sound: 15,
    intrusions: 5,
  };

  // Animate counter updates
  Object.keys(stats).forEach((stat) => {
    const element = document.querySelector(`.stat-count`);
    if (element) {
      animateCounter(element, stats[stat]);
    }
  });
}

function animateCounter(element, target) {
  let current = 0;
  const increment = target / 50;
  const timer = setInterval(() => {
    current += increment;
    if (current >= target) {
      current = target;
      clearInterval(timer);
    }
    element.textContent = Math.floor(current);
  }, 20);
}

function loadCameraFeeds() {
  // Simulate camera feed updates
  const cameras = document.querySelectorAll(".camera-feed");

  cameras.forEach((camera) => {
    const anomaly = camera.getAttribute("data-anomaly");

    if (anomaly !== "none") {
      // Add pulse animation for anomalies
      camera.classList.add("pulse");

      // Add event listener for camera click
      camera.addEventListener("click", function () {
        showCameraDetail(anomaly);
      });
    }
  });
}

function showCameraDetail(anomalyType) {
  const anomalyNames = {
    accident: "Car Accident",
    fire: "Fire Incident",
    weapon: "Weapon Detection",
    fall: "Person Fall",
    sound: "Loud Sound",
    intrusion: "Unauthorized Intrusion",
  };

  alert(
    `Detailed view for ${
      anomalyNames[anomalyType] || "Anomaly"
    }\n\nThis would show live camera feed, incident details, and response options.`
  );
}

function setupRealTimeUpdates() {
  // Simulate real-time updates
  setInterval(() => {
    updateLiveUpdates();
    updateRecentEvents();
  }, 10000); // Update every 10 seconds
}

function updateLiveUpdates() {
  const updates = [
    {
      type: "accident",
      message: "Minor vehicle collision detected in parking area",
      time: "just now",
    },
    {
      type: "sound",
      message: "Loud noise detected in common area - checking...",
      time: "1 minute ago",
    },
    {
      type: "fall",
      message: "Person fall detected - medical team alerted",
      time: "3 minutes ago",
    },
  ];

  const updatesContainer = document.querySelector(".live-updates");
  if (updatesContainer) {
    // Rotate updates
    const randomUpdate = updates[Math.floor(Math.random() * updates.length)];

    const newUpdate = document.createElement("div");
    newUpdate.className = "update-item";
    newUpdate.innerHTML = `
            <div class="update-icon ${randomUpdate.type}">
                <i class="fas ${getAnomalyIcon(randomUpdate.type)}"></i>
            </div>
            <div class="update-content">
                <p><strong>${randomUpdate.message}</strong></p>
                <p class="update-time">${randomUpdate.time}</p>
            </div>
        `;

    updatesContainer.insertBefore(newUpdate, updatesContainer.firstChild);

    // Remove old updates if too many
    if (updatesContainer.children.length > 5) {
      updatesContainer.removeChild(updatesContainer.lastChild);
    }
  }
}

function updateRecentEvents() {
  // Add a new random event to the table
  const eventTypes = [
    "accident",
    "fire",
    "weapon",
    "fall",
    "sound",
    "intrusion",
  ];
  const randomType = eventTypes[Math.floor(Math.random() * eventTypes.length)];

  const newEvent = {
    time: new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
    camera: ["Entrance Gate", "Parking Area", "Warehouse", "Common Area"][
      Math.floor(Math.random() * 4)
    ],
    anomalyType: getAnomalyName(randomType),
    severity: ["high", "medium", "low"][Math.floor(Math.random() * 3)],
    location: [
      "Main Entrance",
      "North Parking",
      "Storage Area",
      "Central Plaza",
    ][Math.floor(Math.random() * 4)],
  };

  const tableBody = document.querySelector("#recentEvents");
  if (tableBody) {
    const newRow = document.createElement("tr");
    newRow.innerHTML = `
            <td>
                <strong>${newEvent.time}</strong><br>
                <small>${new Date().toLocaleDateString()}</small>
            </td>
            <td>${newEvent.camera}</td>
            <td>
                <div class="anomaly-type">
                    <i class="fas ${getAnomalyIcon(newEvent.anomalyType)}"></i>
                    ${newEvent.anomalyType}
                </div>
            </td>
            <td><span class="severity ${
              newEvent.severity
            }">${newEvent.severity.toUpperCase()}</span></td>
            <td>${newEvent.location}</td>
            <td>
                <button class="action-btn view" onclick="viewEvent('${
                  newEvent.anomalyType
                }')">
                    <i class="fas fa-eye"></i> View
                </button>
                <button class="action-btn resolve" onclick="resolveEvent('${
                  newEvent.anomalyType
                }')">
                    <i class="fas fa-check"></i> Resolve
                </button>
            </td>
        `;

    tableBody.insertBefore(newRow, tableBody.firstChild);

    // Remove old events if too many
    if (tableBody.children.length > 10) {
      tableBody.removeChild(tableBody.lastChild);
    }
  }
}

function getAnomalyName(type) {
  const names = {
    accident: "Car Accident",
    fire: "Fire Detected",
    weapon: "Weapon Detected",
    fall: "Person Fall",
    sound: "Loud Sound",
    intrusion: "Unauthorized Intrusion",
  };
  return names[type] || type;
}

function getAnomalyIcon(type) {
  const icons = {
    "Car Accident": "fa-car-crash",
    "Fire Detected": "fa-fire",
    "Weapon Detected": "fa-gun",
    "Person Fall": "fa-person-falling",
    "Loud Sound": "fa-volume-up",
    "Unauthorized Intrusion": "fa-user-secret",
  };
  return icons[type] || "fa-exclamation-triangle";
}

function setupDashboardEventListeners() {
  // Pause/Resume updates button
  const pauseBtn = document.getElementById("pauseUpdatesBtn");
  if (pauseBtn) {
    let updatesPaused = false;

    pauseBtn.addEventListener("click", function () {
      updatesPaused = !updatesPaused;

      if (updatesPaused) {
        this.innerHTML = '<i class="fas fa-play"></i> Resume Updates';
        this.classList.remove("btn-outline");
        this.classList.add("btn-primary");
      } else {
        this.innerHTML = '<i class="fas fa-pause"></i> Pause Updates';
        this.classList.remove("btn-primary");
        this.classList.add("btn-outline");
      }

      showNotification(updatesPaused ? "Updates paused" : "Updates resumed");
    });
  }

  // Export button
  const exportBtn = document.getElementById("exportBtn");
  if (exportBtn) {
    exportBtn.addEventListener("click", function () {
      showNotification("Exporting dashboard data...");

      // Simulate export process
      setTimeout(() => {
        showNotification("Dashboard data exported successfully", "success");
      }, 1500);
    });
  }

  // Camera view toggle
  const gridViewBtn = document.querySelector('[class*="grid-view"]');
  const listViewBtn = document.querySelector('[class*="list-view"]');

  if (gridViewBtn && listViewBtn) {
    gridViewBtn.addEventListener("click", function () {
      document.querySelector(".camera-grid").style.display = "grid";
      this.classList.add("active");
      listViewBtn.classList.remove("active");
    });

    listViewBtn.addEventListener("click", function () {
      document.querySelector(".camera-grid").style.display = "block";
      this.classList.add("active");
      gridViewBtn.classList.remove("active");
    });
  }
}

// Global functions for dashboard
function viewEvent(eventType) {
  showNotification(`Opening details for ${eventType}`);
  // In a real app, this would navigate to event details
}

function resolveEvent(eventType) {
  showNotification(`${eventType} marked as resolved`, "success");
  // In a real app, this would update event status in database
}
