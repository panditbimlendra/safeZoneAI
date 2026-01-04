// Shared functionality across all pages

// Sample events data for dashboard
const recentEvents = [
  {
    id: 1,
    time: "10:23 AM",
    timestamp: "2023-10-15 10:23:00",
    camera: "Parking Area",
    anomalyType: "Car Accident",
    severity: "high",
    location: "North Parking",
    status: "active",
  },
  {
    id: 2,
    time: "09:45 AM",
    timestamp: "2023-10-15 09:45:00",
    camera: "Warehouse",
    anomalyType: "Fire Detected",
    severity: "high",
    location: "Storage Area",
    status: "resolved",
  },
  {
    id: 3,
    time: "08:12 AM",
    timestamp: "2023-10-15 08:12:00",
    camera: "Common Area",
    anomalyType: "Loud Sound",
    severity: "medium",
    location: "Central Plaza",
    status: "active",
  },
  {
    id: 4,
    time: "07:30 AM",
    timestamp: "2023-10-15 07:30:00",
    camera: "Entrance Gate",
    anomalyType: "Weapon Detected",
    severity: "high",
    location: "Main Entrance",
    status: "resolved",
  },
  {
    id: 5,
    time: "06:45 AM",
    timestamp: "2023-10-15 06:45:00",
    camera: "Hallway",
    anomalyType: "Person Fall",
    severity: "medium",
    location: "First Floor",
    status: "active",
  },
];

// Initialize on page load
document.addEventListener("DOMContentLoaded", function () {
  updateCurrentTime();
  setupNavigation();
  setupNotificationSystem();
  initializePageSpecificFeatures();

  // Check if we're on dashboard to load events
  if (document.getElementById("recentEvents")) {
    loadRecentEvents();
  }
});

// Update current time in footer
function updateCurrentTime() {
  const now = new Date();
  const timeString = now.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const timeElement = document.getElementById("currentTime");
  if (timeElement) {
    timeElement.textContent = timeString;
  }

  // Update every second
  setTimeout(updateCurrentTime, 1000);
}

// Setup navigation active states
function setupNavigation() {
  // Set active nav item based on current page
  const currentPage = window.location.pathname.split("/").pop() || "index.html";
  const navLinks = document.querySelectorAll(".nav-links a");

  navLinks.forEach((link) => {
    const href = link.getAttribute("href");
    if (
      href === currentPage ||
      (currentPage === "index.html" && href.includes("index")) ||
      (currentPage.includes("dashboard") && href.includes("dashboard")) ||
      (currentPage.includes("events") && href.includes("events")) ||
      (currentPage.includes("analytics") && href.includes("analytics")) ||
      (currentPage.includes("alerts") && href.includes("alerts")) ||
      (currentPage.includes("settings") && href.includes("settings"))
    ) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  // User menu toggle
  const userInfo = document.querySelector(".user-info");
  if (userInfo) {
    userInfo.addEventListener("click", function () {
      showUserMenu();
    });
  }
}

function showUserMenu() {
  // Create user menu if it doesn't exist
  let userMenu = document.querySelector(".user-menu");

  if (!userMenu) {
    userMenu = document.createElement("div");
    userMenu.className = "user-menu";
    userMenu.innerHTML = `
            <div class="menu-header">
                <img src="https://ui-avatars.com/api/?name=Admin+User&background=1a73e8&color=fff" alt="User">
                <div>
                    <h4>Admin User</h4>
                    <p>Administrator</p>
                </div>
            </div>
            <div class="menu-items">
                <a href="#"><i class="fas fa-user"></i> Profile</a>
                <a href="#"><i class="fas fa-cog"></i> Account Settings</a>
                <a href="#"><i class="fas fa-moon"></i> Dark Mode</a>
                <div class="divider"></div>
                <a href="#" class="logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </div>
        `;

    // Add styles
    const style = document.createElement("style");
    style.textContent = `
            .user-menu {
                position: absolute;
                top: 70px;
                right: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                width: 250px;
                z-index: 1000;
                animation: fadeIn 0.2s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .menu-header {
                padding: 20px;
                display: flex;
                align-items: center;
                gap: 15px;
                border-bottom: 1px solid var(--gray-light);
            }
            
            .menu-header img {
                width: 50px;
                height: 50px;
                border-radius: 50%;
            }
            
            .menu-header h4 {
                margin: 0;
                font-size: 16px;
            }
            
            .menu-header p {
                margin: 5px 0 0;
                color: var(--gray-dark);
                font-size: 14px;
            }
            
            .menu-items {
                padding: 10px 0;
            }
            
            .menu-items a {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 12px 20px;
                color: var(--dark-color);
                text-decoration: none;
                transition: background 0.2s;
            }
            
            .menu-items a:hover {
                background: rgba(26, 115, 232, 0.1);
                color: var(--primary-color);
            }
            
            .menu-items a i {
                width: 20px;
                text-align: center;
            }
            
            .menu-items .divider {
                height: 1px;
                background: var(--gray-light);
                margin: 10px 0;
            }
            
            .menu-items .logout {
                color: var(--danger-color);
            }
            
            .menu-items .logout:hover {
                background: rgba(234, 67, 53, 0.1);
            }
        `;
    document.head.appendChild(style);

    document.body.appendChild(userMenu);

    // Close menu when clicking outside
    document.addEventListener("click", function closeMenu(e) {
      if (!userMenu.contains(e.target) && !userInfo.contains(e.target)) {
        userMenu.remove();
        document.removeEventListener("click", closeMenu);
      }
    });
  } else {
    userMenu.remove();
  }
}

// Setup notification system
function setupNotificationSystem() {
  // Add notification container to body
  const notificationContainer = document.createElement("div");
  notificationContainer.id = "notification-container";
  document.body.appendChild(notificationContainer);

  // Add styles for notifications
  const style = document.createElement("style");
  style.textContent = `
        #notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        }
        
        .notification {
            background: white;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 300px;
            max-width: 400px;
            animation: slideIn 0.3s ease;
            border-left: 4px solid var(--primary-color);
        }
        
        .notification.success {
            border-left-color: var(--success-color);
        }
        
        .notification.warning {
            border-left-color: var(--warning-color);
        }
        
        .notification.danger {
            border-left-color: var(--danger-color);
        }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .notification i {
            font-size: 18px;
        }
        
        .notification.success i {
            color: var(--success-color);
        }
        
        .notification.warning i {
            color: var(--warning-color);
        }
        
        .notification.danger i {
            color: var(--danger-color);
        }
        
        .notification span {
            flex: 1;
            font-size: 14px;
        }
        
        .notification button {
            background: none;
            border: none;
            color: var(--gray-dark);
            cursor: pointer;
            padding: 5px;
        }
        
        .notification button:hover {
            color: var(--dark-color);
        }
    `;
  document.head.appendChild(style);
}

// Show notification
function showNotification(message, type = "info") {
  const container = document.getElementById("notification-container");
  if (!container) return;

  const notification = document.createElement("div");
  notification.className = `notification ${type}`;

  const icon =
    type === "success"
      ? "fa-check-circle"
      : type === "warning"
      ? "fa-exclamation-triangle"
      : type === "danger"
      ? "fa-times-circle"
      : "fa-info-circle";

  notification.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;

  container.appendChild(notification);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 5000);
}

// Initialize page-specific features
function initializePageSpecificFeatures() {
  // Common button handlers
  setupCommonButtons();

  // Update system status
  updateSystemStatus();
}

function setupCommonButtons() {
  // Refresh buttons
  document
    .querySelectorAll('#refreshBtn, .btn:contains("Refresh")')
    .forEach((btn) => {
      if (!btn.id || btn.id === "refreshBtn") {
        btn.addEventListener("click", function () {
          this.classList.add("rotating");
          setTimeout(() => {
            this.classList.remove("rotating");
            showNotification("Page refreshed successfully");
            location.reload();
          }, 1000);
        });
      }
    });

  // Add rotating animation style
  const style = document.createElement("style");
  style.textContent = `
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .rotating {
            animation: rotate 1s linear;
        }
    `;
  document.head.appendChild(style);
}

function updateSystemStatus() {
  const statusElement = document.querySelector(".system-status");
  if (statusElement) {
    const status = Math.random() > 0.1 ? "Operational" : "Degraded Performance";
    const color =
      status === "Operational"
        ? "var(--success-color)"
        : "var(--warning-color)";

    const statusSpan = statusElement.querySelector(".status");
    if (statusSpan) {
      statusSpan.textContent = status;
      statusSpan.style.color = color;
    }
  }
}

// Dashboard-specific functions
function loadRecentEvents() {
  const eventsTable = document.getElementById("recentEvents");
  if (!eventsTable) return;

  eventsTable.innerHTML = "";

  recentEvents.forEach((event) => {
    const row = document.createElement("tr");
    row.innerHTML = `
            <td>
                <strong>${event.time}</strong><br>
                <small>${event.timestamp.split(" ")[0]}</small>
            </td>
            <td>${event.camera}</td>
            <td>
                <div class="anomaly-type">
                    <i class="fas ${getAnomalyIcon(event.anomalyType)}"></i>
                    ${event.anomalyType}
                </div>
            </td>
            <td><span class="severity ${
              event.severity
            }">${event.severity.toUpperCase()}</span></td>
            <td>${event.location}</td>
            <td>
                <button class="action-btn view" onclick="viewEventDetail('${
                  event.anomalyType
                }')">
                    <i class="fas fa-eye"></i> View
                </button>
                <button class="action-btn resolve" onclick="resolveEventDetail(${
                  event.id
                })">
                    <i class="fas fa-check"></i> Resolve
                </button>
            </td>
        `;
    eventsTable.appendChild(row);
  });
}

function getAnomalyIcon(type) {
  const icons = {
    "Car Accident": "fa-car-crash",
    "Fire Detected": "fa-fire",
    "Loud Sound": "fa-volume-up",
    "Weapon Detected": "fa-gun",
    "Person Fall": "fa-person-falling",
    Intrusion: "fa-user-secret",
    "Unattended Object": "fa-suitcase",
  };
  return icons[type] || "fa-exclamation-triangle";
}

function viewEventDetail(eventType) {
  showNotification(`Opening details for ${eventType}`);
  // In a real app, navigate to event details
}

function resolveEventDetail(eventId) {
  const event = recentEvents.find((e) => e.id === eventId);
  if (event) {
    event.status = "resolved";
    loadRecentEvents();
    showNotification(`Event #${eventId} marked as resolved`, "success");
  }
}

// Simulate live updates for dashboard
function simulateLiveUpdates() {
  // Update time every second
  setInterval(updateCurrentTime, 1000);

  // Simulate new events every 30 seconds
  setInterval(() => {
    const eventTypes = [
      "Car Accident",
      "Loud Sound",
      "Person Fall",
      "Intrusion",
    ];
    const randomType =
      eventTypes[Math.floor(Math.random() * eventTypes.length)];
    const randomCamera = [
      "Parking Area",
      "Common Area",
      "Entrance Gate",
      "Warehouse",
    ][Math.floor(Math.random() * 4)];
    const randomSeverity =
      Math.random() > 0.7 ? "high" : Math.random() > 0.5 ? "medium" : "low";

    const newEvent = {
      id: recentEvents.length + 1,
      time: new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      timestamp: new Date().toISOString().replace("T", " ").substring(0, 19),
      camera: randomCamera,
      anomalyType: randomType,
      severity: randomSeverity,
      location:
        randomCamera === "Parking Area"
          ? "North Parking"
          : randomCamera === "Warehouse"
          ? "Storage Area"
          : randomCamera === "Common Area"
          ? "Central Plaza"
          : "Main Entrance",
      status: "active",
    };

    recentEvents.unshift(newEvent);
    if (recentEvents.length > 10) recentEvents.pop();

    // Update table if on dashboard
    if (document.getElementById("recentEvents")) {
      loadRecentEvents();
    }

    // Show notification for high severity events
    if (randomSeverity === "high") {
      showNotification(
        `New ${randomType} detected at ${randomCamera}`,
        "warning"
      );
    }
  }, 30000);
}

// Initialize live updates if on dashboard
if (document.querySelector(".dashboard")) {
  simulateLiveUpdates();
}

// Export functions for use in HTML onclick attributes
window.viewEvent = viewEventDetail;
window.resolveEvent = resolveEventDetail;
window.showNotification = showNotification;
