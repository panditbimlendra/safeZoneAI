// Alerts page specific functionality
document.addEventListener("DOMContentLoaded", function () {
  initializeAlertsPage();
  setupAlertEventListeners();
});

function initializeAlertsPage() {
  updateAlertCounts();
  setupPriorityAlerts();
  initializeAlertFilters();
}

function updateAlertCounts() {
  // Update active alerts count
  const activeAlerts = document.querySelectorAll(".alert-card").length;
  document.getElementById("activeAlertsCount").textContent = activeAlerts;

  // Update last alert time
  const lastAlertTime = document.querySelector(".alert-time");
  if (lastAlertTime) {
    document.getElementById("lastAlertTime").textContent =
      lastAlertTime.textContent;
  }
}

function setupPriorityAlerts() {
  // Add click handlers to priority alerts
  document.querySelectorAll(".alert-card").forEach((card) => {
    card.addEventListener("click", function (e) {
      // Don't trigger if clicking action buttons
      if (!e.target.closest(".alert-actions")) {
        const title = this.querySelector("h4").textContent;
        showNotification(`Opening details for: ${title}`);
      }
    });
  });

  // Add hover effects
  document.querySelectorAll(".alert-card").forEach((card) => {
    card.addEventListener("mouseenter", function () {
      this.style.transform = "translateY(-2px)";
      this.style.boxShadow = "0 6px 20px rgba(0,0,0,0.1)";
    });

    card.addEventListener("mouseleave", function () {
      this.style.transform = "";
      this.style.boxShadow = "";
    });
  });
}

function initializeAlertFilters() {
  // Filter buttons
  document.querySelectorAll(".alert-filters .btn-sm").forEach((btn) => {
    btn.addEventListener("click", function () {
      // Update active state
      document.querySelectorAll(".alert-filters .btn-sm").forEach((b) => {
        b.classList.remove("active");
      });
      this.classList.add("active");

      // Apply filter
      const filter = this.textContent.toLowerCase();
      filterAlerts(filter);
    });
  });

  // Load more alerts button
  const loadMoreBtn = document.getElementById("loadMoreAlerts");
  if (loadMoreBtn) {
    loadMoreBtn.addEventListener("click", loadMoreAlerts);
  }
}

function filterAlerts(filter) {
  const alerts = document.querySelectorAll(".alert-item");

  alerts.forEach((alert) => {
    switch (filter) {
      case "unread":
        alert.style.display = "flex";
        // In real app, would check if alert is unread
        break;
      case "resolved":
        alert.style.display = "none";
        // In real app, would check if alert is resolved
        break;
      default:
        alert.style.display = "flex";
    }
  });

  showNotification(`Showing ${filter} alerts`);
}

function loadMoreAlerts() {
  const recentAlerts = document.querySelector(".recent-alerts");
  if (!recentAlerts) return;

  // Sample additional alerts
  const newAlerts = [
    {
      type: "accident",
      icon: "fa-car-crash",
      title: "Minor Collision",
      description: "Two vehicles in parking lot",
      time: "8 hours ago",
      severity: "low",
    },
    {
      type: "sound",
      icon: "fa-volume-up",
      title: "Loud Noise",
      description: "Possible equipment issue",
      time: "10 hours ago",
      severity: "medium",
    },
  ];

  newAlerts.forEach((alert) => {
    const alertItem = document.createElement("div");
    alertItem.className = "alert-item";
    alertItem.innerHTML = `
            <div class="alert-type">
                <div class="alert-type-icon ${alert.type}">
                    <i class="fas ${alert.icon}"></i>
                </div>
                <div class="alert-info">
                    <h5>${alert.title}</h5>
                    <p>${alert.description}</p>
                </div>
            </div>
            <div class="alert-meta">
                <span class="alert-time">${alert.time}</span>
                <span class="severity ${alert.severity}">${alert.severity}</span>
            </div>
        `;

    recentAlerts.appendChild(alertItem);
  });

  showNotification("Loaded more alerts", "success");
}

function setupAlertEventListeners() {
  // Mark all as read button
  const markAllReadBtn = document.getElementById("markAllRead");
  if (markAllReadBtn) {
    markAllReadBtn.addEventListener("click", function () {
      document.querySelectorAll(".alert-card").forEach((card) => {
        card.classList.add("read");
      });

      // Update badge count
      const badge = document.querySelector(".nav-links .badge");
      if (badge) {
        badge.style.display = "none";
      }

      showNotification("All alerts marked as read", "success");
    });
  }

  // Clear all alerts button
  const clearAllBtn = document.getElementById("clearAllAlerts");
  if (clearAllBtn) {
    clearAllBtn.addEventListener("click", function () {
      if (confirm("Clear all alerts? This action cannot be undone.")) {
        document.querySelectorAll(".alert-card").forEach((card) => {
          card.remove();
        });

        document.querySelectorAll(".alert-item").forEach((item) => {
          item.remove();
        });

        updateAlertCounts();
        showNotification("All alerts cleared", "success");
      }
    });
  }

  // Save preferences button
  const savePrefsBtn = document.getElementById("savePreferences");
  if (savePrefsBtn) {
    savePrefsBtn.addEventListener("click", function () {
      saveAlertPreferences();
    });
  }

  // Alert schedule select
  const alertHoursSelect = document.getElementById("alertHours");
  if (alertHoursSelect) {
    alertHoursSelect.addEventListener("change", function () {
      const quietHours = document.querySelector(
        ".preference-item:nth-child(3)"
      );
      if (this.value === "custom") {
        quietHours.style.display = "block";
      } else {
        quietHours.style.display = "none";
      }
    });
  }
}

// Alert action functions (called from HTML onclick)
function acknowledgeAlert(alertId) {
  const alertCard = document.querySelector(`.alert-card:nth-child(${alertId})`);
  if (alertCard) {
    alertCard.classList.add("acknowledged");

    const btn = alertCard.querySelector(".alert-actions .btn-outline");
    if (btn) {
      btn.innerHTML = '<i class="fas fa-check-circle"></i> Acknowledged';
      btn.disabled = true;
    }

    showNotification("Alert acknowledged", "success");
    updateAlertCounts();
  }
}

function resolveAlert(alertId) {
  const alertCard = document.querySelector(`.alert-card:nth-child(${alertId})`);
  if (alertCard) {
    // Add resolved styling
    alertCard.classList.add("resolved");
    alertCard.style.opacity = "0.7";

    // Update buttons
    const actions = alertCard.querySelector(".alert-actions");
    if (actions) {
      actions.innerHTML = `
                <button class="btn-sm btn-outline" onclick="reopenAlert(${alertId})">
                    <i class="fas fa-redo"></i> Reopen
                </button>
                <button class="btn-sm btn-outline" onclick="deleteAlert(${alertId})">
                    <i class="fas fa-trash"></i> Delete
                </button>
            `;
    }

    // Update status in body
    const body = alertCard.querySelector(".alert-body");
    if (body) {
      const statusRow = document.createElement("p");
      statusRow.innerHTML = "<strong>Status:</strong> Resolved";
      body.appendChild(statusRow);
    }

    showNotification("Alert resolved", "success");
    updateAlertCounts();
  }
}

function reopenAlert(alertId) {
  const alertCard = document.querySelector(`.alert-card:nth-child(${alertId})`);
  if (alertCard) {
    alertCard.classList.remove("resolved");
    alertCard.style.opacity = "1";

    // Restore original buttons
    const actions = alertCard.querySelector(".alert-actions");
    if (actions) {
      actions.innerHTML = `
                <button class="btn-sm btn-outline" onclick="acknowledgeAlert(${alertId})">
                    <i class="fas fa-check"></i> Acknowledge
                </button>
                <button class="btn-sm btn-danger" onclick="resolveAlert(${alertId})">
                    <i class="fas fa-times"></i> Resolve
                </button>
            `;
    }

    // Remove status from body
    const body = alertCard.querySelector(".alert-body");
    if (body) {
      const statusRow = body.querySelector('p:contains("Status")');
      if (statusRow) {
        statusRow.remove();
      }
    }

    showNotification("Alert reopened");
    updateAlertCounts();
  }
}

function deleteAlert(alertId) {
  const alertCard = document.querySelector(`.alert-card:nth-child(${alertId})`);
  if (alertCard && confirm("Delete this alert permanently?")) {
    alertCard.remove();
    showNotification("Alert deleted", "success");
    updateAlertCounts();
  }
}

function saveAlertPreferences() {
  const preferences = {
    email: document.getElementById("prefEmail").checked,
    sms: document.getElementById("prefSMS").checked,
    push: document.getElementById("prefPush").checked,
    sound: document.getElementById("prefSound").checked,
    schedule: document.getElementById("alertHours").value,
    quietStart: document.getElementById("quietStart").value,
    quietEnd: document.getElementById("quietEnd").value,
  };

  // Save to localStorage
  localStorage.setItem("alert-preferences", JSON.stringify(preferences));

  showNotification("Alert preferences saved", "success");
}

// Add CSS for alert states
const alertStyles = document.createElement("style");
alertStyles.textContent = `
    .alert-card.read {
        opacity: 0.7;
        border-left-color: var(--gray-light);
    }
    
    .alert-card.resolved {
        border-left-color: var(--success-color);
    }
    
    .alert-card.acknowledged {
        border-left-color: var(--primary-color);
    }
    
    .alert-card.resolved .alert-icon {
        background-color: var(--success-color) !important;
    }
    
    .alert-card.acknowledged .alert-icon {
        background-color: var(--primary-color) !important;
    }
    
    .severity.critical {
        background-color: rgba(234, 67, 53, 0.1);
        color: var(--danger-color);
    }
    
    .severity.high {
        background-color: rgba(251, 188, 5, 0.1);
        color: var(--warning-color);
    }
    
    .severity.medium {
        background-color: rgba(66, 133, 244, 0.1);
        color: var(--primary-color);
    }
    
    .severity.low {
        background-color: rgba(52, 168, 83, 0.1);
        color: var(--success-color);
    }
`;
document.head.appendChild(alertStyles);
