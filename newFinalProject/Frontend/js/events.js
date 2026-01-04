// Events page specific functionality
document.addEventListener("DOMContentLoaded", function () {
  initializeEventsPage();
  loadEventsData();
  setupEventListeners();
});

// Sample events data
const eventsData = [
  {
    id: 1,
    timestamp: "2023-10-15 10:23:00",
    type: "accident",
    title: "Car Accident",
    description: "Two-vehicle collision in parking area",
    camera: "Parking Area",
    location: "North Parking, Section B",
    severity: "high",
    confidence: 89.3,
    status: "active",
    image: "accident.jpg",
  },
  {
    id: 2,
    timestamp: "2023-10-15 09:45:00",
    type: "fire",
    title: "Fire Detected",
    description: "Small fire in warehouse storage area",
    camera: "Warehouse",
    location: "Storage Area A",
    severity: "critical",
    confidence: 96.7,
    status: "resolved",
    image: "fire.jpg",
  },
  {
    id: 3,
    timestamp: "2023-10-15 08:12:00",
    type: "sound",
    title: "Loud Sound",
    description: "Abnormally loud noise in common area",
    camera: "Common Area",
    location: "Central Plaza",
    severity: "medium",
    confidence: 78.5,
    status: "active",
    image: "sound.jpg",
  },
  {
    id: 4,
    timestamp: "2023-10-14 15:30:00",
    type: "weapon",
    title: "Weapon Detected",
    description: "Firearm detected at entrance security",
    camera: "Entrance Gate",
    location: "Main Entrance",
    severity: "high",
    confidence: 92.1,
    status: "resolved",
    image: "weapon.jpg",
  },
  {
    id: 5,
    timestamp: "2023-10-14 11:20:00",
    type: "fall",
    title: "Person Fall",
    description: "Elderly person fell in common area",
    camera: "Common Area",
    location: "Seating Area",
    severity: "medium",
    confidence: 85.2,
    status: "resolved",
    image: "fall.jpg",
  },
  {
    id: 6,
    timestamp: "2023-10-14 09:15:00",
    type: "intrusion",
    title: "Unauthorized Access",
    description: "After hours access to restricted area",
    camera: "Warehouse",
    location: "Back Entrance",
    severity: "high",
    confidence: 88.7,
    status: "active",
    image: "intrusion.jpg",
  },
];

function initializeEventsPage() {
  updateEventsCount();
  setupFilterListeners();
}

function loadEventsData() {
  const container = document.getElementById("eventsContainer");
  if (!container) return;

  container.innerHTML = "";

  eventsData.forEach((event) => {
    const eventCard = createEventCard(event);
    container.appendChild(eventCard);
  });
}

function createEventCard(event) {
  const card = document.createElement("div");
  card.className = "event-card";
  card.dataset.id = event.id;
  card.dataset.type = event.type;
  card.dataset.severity = event.severity;
  card.dataset.status = event.status;

  const severityClass = event.severity === "critical" ? "high" : event.severity;

  card.innerHTML = `
        <div class="event-header">
            <div class="event-type">
                <div class="event-type-icon ${event.type}">
                    <i class="fas ${getEventIcon(event.type)}"></i>
                </div>
                <div class="event-type-info">
                    <h4>${event.title}</h4>
                    <span class="severity ${severityClass}">${event.severity.toUpperCase()}</span>
                </div>
            </div>
        </div>
        
        <div class="event-body">
            <div class="event-image">
                <i class="fas ${getEventIcon(event.type)}"></i>
            </div>
            
            <div class="event-details">
                <div class="event-detail">
                    <i class="fas fa-camera"></i>
                    <span>${event.camera}</span>
                </div>
                <div class="event-detail">
                    <i class="fas fa-map-marker-alt"></i>
                    <span>${event.location}</span>
                </div>
                <div class="event-detail">
                    <i class="fas fa-bullseye"></i>
                    <span>Confidence: ${event.confidence}%</span>
                </div>
                <div class="event-detail">
                    <i class="fas fa-clock"></i>
                    <span>${formatTime(event.timestamp)}</span>
                </div>
            </div>
            
            <p>${event.description}</p>
        </div>
        
        <div class="event-footer">
            <span class="event-status ${event.status}">
                ${event.status === "active" ? "Needs Attention" : "Resolved"}
            </span>
            <div class="event-actions">
                <button class="btn-sm btn-outline" onclick="viewEventDetails(${
                  event.id
                })">
                    <i class="fas fa-eye"></i> View
                </button>
                <button class="btn-sm ${
                  event.status === "active" ? "btn-primary" : "btn-outline"
                }" 
                        onclick="${
                          event.status === "active"
                            ? `resolveEvent(${event.id})`
                            : `reopenEvent(${event.id})`
                        }">
                    <i class="fas ${
                      event.status === "active" ? "fa-check" : "fa-redo"
                    }"></i>
                    ${event.status === "active" ? "Resolve" : "Reopen"}
                </button>
            </div>
        </div>
    `;

  return card;
}

function getEventIcon(type) {
  const icons = {
    accident: "fa-car-crash",
    fire: "fa-fire",
    weapon: "fa-gun",
    fall: "fa-person-falling",
    sound: "fa-volume-up",
    intrusion: "fa-user-secret",
  };
  return icons[type] || "fa-exclamation-triangle";
}

function formatTime(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 60) {
    return `${diffMins} minutes ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hours ago`;
  } else if (diffDays === 1) {
    return "Yesterday";
  } else {
    return `${diffDays} days ago`;
  }
}

function updateEventsCount() {
  const countElement = document.getElementById("eventsCount");
  if (countElement) {
    countElement.textContent = eventsData.length;
  }
}

function setupFilterListeners() {
  const applyBtn = document.getElementById("applyFiltersBtn");
  const clearBtn = document.getElementById("clearFiltersBtn");

  if (applyBtn) {
    applyBtn.addEventListener("click", applyFilters);
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", clearFilters);
  }

  // Grid/List view toggle
  const gridBtn = document.getElementById("gridViewBtn");
  const listBtn = document.getElementById("listViewBtn");

  if (gridBtn && listBtn) {
    gridBtn.addEventListener("click", () => toggleView("grid"));
    listBtn.addEventListener("click", () => toggleView("list"));
  }

  // Export buttons
  document.querySelectorAll(".export-btn").forEach((btn) => {
    btn.addEventListener("click", function () {
      const format = this.id.replace("export", "").toLowerCase();
      exportEvents(format);
    });
  });

  // Timeline toggle
  const timelineBtn = document.getElementById("toggleTimeline");
  if (timelineBtn) {
    timelineBtn.addEventListener("click", function () {
      const timeline = document.getElementById("timelineContainer");
      if (timeline.style.display === "none") {
        timeline.style.display = "block";
        this.innerHTML = '<i class="fas fa-times"></i> Hide Timeline';
      } else {
        timeline.style.display = "none";
        this.innerHTML = '<i class="fas fa-clock"></i> Show Timeline';
      }
    });
  }
}

function applyFilters() {
  const typeFilter = document.getElementById("eventType").value;
  const severityFilter = document.getElementById("severity").value;
  const cameraFilter = document.getElementById("camera").value;
  const statusFilter = document.getElementById("status").value;
  const startDate = document.getElementById("startDate").value;
  const endDate = document.getElementById("endDate").value;

  const filteredEvents = eventsData.filter((event) => {
    // Type filter
    if (typeFilter !== "all" && event.type !== typeFilter) return false;

    // Severity filter
    if (severityFilter !== "all" && event.severity !== severityFilter) {
      if (
        severityFilter === "high" &&
        event.severity !== "high" &&
        event.severity !== "critical"
      )
        return false;
    }

    // Camera filter
    if (cameraFilter !== "all") {
      const cameraMap = {
        entrance: "Entrance Gate",
        parking: "Parking Area",
        warehouse: "Warehouse",
        common: "Common Area",
      };
      if (event.camera !== cameraMap[cameraFilter]) return false;
    }

    // Status filter
    if (statusFilter !== "all" && event.status !== statusFilter) return false;

    // Date filter
    if (startDate) {
      const eventDate = new Date(event.timestamp).toISOString().split("T")[0];
      if (eventDate < startDate) return false;
    }

    if (endDate) {
      const eventDate = new Date(event.timestamp).toISOString().split("T")[0];
      if (eventDate > endDate) return false;
    }

    return true;
  });

  displayFilteredEvents(filteredEvents);
  showNotification(`Found ${filteredEvents.length} events matching filters`);
}

function clearFilters() {
  document.getElementById("eventType").value = "all";
  document.getElementById("severity").value = "all";
  document.getElementById("camera").value = "all";
  document.getElementById("status").value = "all";
  document.getElementById("startDate").value = "";
  document.getElementById("endDate").value = "";

  loadEventsData();
  showNotification("Filters cleared");
}

function displayFilteredEvents(filteredEvents) {
  const container = document.getElementById("eventsContainer");
  if (!container) return;

  container.innerHTML = "";

  if (filteredEvents.length === 0) {
    container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <h3>No events found</h3>
                <p>Try adjusting your filters to see more results</p>
                <button class="btn btn-outline" onclick="clearFilters()">
                    Clear Filters
                </button>
            </div>
        `;
    return;
  }

  filteredEvents.forEach((event) => {
    const eventCard = createEventCard(event);
    container.appendChild(eventCard);
  });
}

function toggleView(viewType) {
  const container = document.getElementById("eventsContainer");
  const gridBtn = document.getElementById("gridViewBtn");
  const listBtn = document.getElementById("listViewBtn");

  if (viewType === "list") {
    container.classList.add("list-view");
    container.classList.remove("grid-view");
    gridBtn.classList.remove("active");
    listBtn.classList.add("active");
  } else {
    container.classList.add("grid-view");
    container.classList.remove("list-view");
    gridBtn.classList.add("active");
    listBtn.classList.remove("active");
  }
}

function exportEvents(format) {
  showNotification(`Exporting events as ${format.toUpperCase()}...`);

  // Simulate export process
  setTimeout(() => {
    let content = "";

    switch (format) {
      case "csv":
        content = "Time,Type,Description,Severity,Location,Status\n";
        eventsData.forEach((event) => {
          content += `${event.timestamp},${event.type},${event.description},${event.severity},${event.location},${event.status}\n`;
        });
        break;

      case "json":
        content = JSON.stringify(eventsData, null, 2);
        break;

      default:
        content = `Events Export - ${new Date().toLocaleString()}\n\n`;
        eventsData.forEach((event) => {
          content += `Time: ${event.timestamp}\nType: ${event.type}\nDescription: ${event.description}\nSeverity: ${event.severity}\nLocation: ${event.location}\nStatus: ${event.status}\n\n`;
        });
    }

    // Create download link
    const blob = new Blob([content], { type: `text/${format}` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `events_export_${
      new Date().toISOString().split("T")[0]
    }.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification(
      `Events exported successfully as ${format.toUpperCase()}`,
      "success"
    );
  }, 1000);
}

// Event action functions
function viewEventDetails(eventId) {
  const event = eventsData.find((e) => e.id === eventId);
  if (event) {
    showNotification(`Opening detailed view for ${event.title}`);
    // In real app, navigate to event detail page or show modal
  }
}

function resolveEvent(eventId) {
  const event = eventsData.find((e) => e.id === eventId);
  if (event) {
    event.status = "resolved";
    loadEventsData();
    showNotification(`${event.title} marked as resolved`, "success");
  }
}

function reopenEvent(eventId) {
  const event = eventsData.find((e) => e.id === eventId);
  if (event) {
    event.status = "active";
    loadEventsData();
    showNotification(`${event.title} reopened for review`);
  }
}

function setupEventListeners() {
  // New event button
  const newEventBtn = document.getElementById("newEventBtn");
  if (newEventBtn) {
    newEventBtn.addEventListener("click", function () {
      showNotification("Opening new event creation form");
      // In real app, show modal or navigate to create event page
    });
  }

  // Bulk actions button
  const bulkActionsBtn = document.getElementById("bulkActionsBtn");
  if (bulkActionsBtn) {
    bulkActionsBtn.addEventListener("click", function () {
      showNotification("Bulk actions menu opened");
      // In real app, show bulk actions menu
    });
  }
}
