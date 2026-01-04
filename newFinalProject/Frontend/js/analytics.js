// Analytics page specific functionality
document.addEventListener("DOMContentLoaded", function () {
  initializeAnalytics();
  setupAnalyticsEventListeners();
});

function initializeAnalytics() {
  updateAnalyticsData();
  setupChartPlaceholders();
  updateTimeAnalysis();
}

function updateAnalyticsData() {
  // Update metrics with random data for demonstration
  const metrics = document.querySelectorAll(".metric-value");
  metrics.forEach((metric) => {
    const current = parseFloat(metric.textContent);
    const change = (Math.random() - 0.5) * 0.1; // ±5% change
    const newValue = current * (1 + change);

    // Update display with animation
    animateMetric(metric, newValue);
  });
}

function animateMetric(element, target) {
  let current = parseFloat(element.textContent);
  const increment = (target - current) / 20;
  let step = 0;

  const timer = setInterval(() => {
    current += increment;
    step++;

    if (step >= 20) {
      current = target;
      clearInterval(timer);
    }

    if (element.classList.contains("percent")) {
      element.textContent = current.toFixed(1) + "%";
    } else if (element.classList.contains("time")) {
      element.textContent = current.toFixed(1) + "s";
    } else {
      element.textContent = Math.round(current);
    }
  }, 50);
}

function setupChartPlaceholders() {
  // Add click handlers to chart placeholders
  document.querySelectorAll(".chart-placeholder").forEach((chart) => {
    chart.addEventListener("click", function () {
      const header = this.parentElement.querySelector("h3").textContent;
      showNotification(`Would open detailed ${header} chart view`);
    });
  });

  // Add hover effects to heat map cells
  document.querySelectorAll(".heat-map-cell").forEach((cell) => {
    cell.addEventListener("mouseenter", function () {
      const riskLevel = this.classList.contains("high")
        ? "High Risk"
        : this.classList.contains("medium")
        ? "Medium Risk"
        : "Low Risk";
      this.title = `Area ${this.textContent}: ${riskLevel} (${
        Math.floor(Math.random() * 20) + 1
      } events)`;
    });
  });

  // Setup type distribution items
  document.querySelectorAll(".type-item").forEach((item) => {
    item.addEventListener("click", function () {
      const typeName = this.querySelector("h4").textContent;
      showNotification(`Would show detailed analytics for ${typeName}`);
    });
  });
}

function updateTimeAnalysis() {
  // Update time slots with random data
  document.querySelectorAll(".time-slot").forEach((slot) => {
    const countElement = slot.querySelector(".time-count");
    if (countElement) {
      const current = parseInt(countElement.textContent);
      const newCount = Math.max(
        1,
        Math.floor(current * (0.8 + Math.random() * 0.4))
      );

      // Add animation effect
      countElement.style.transform = "scale(1.2)";
      setTimeout(() => {
        countElement.textContent = newCount;
        countElement.style.transform = "scale(1)";
      }, 300);
    }
  });
}

function setupAnalyticsEventListeners() {
  // Period selector
  const periodSelect = document.getElementById("analyticsPeriod");
  if (periodSelect) {
    periodSelect.addEventListener("change", function () {
      showNotification(
        `Loading analytics for ${this.value.replace("This ", "")}`
      );
      updateAnalyticsForPeriod(this.value);
    });
  }

  // Chart period selectors
  document.querySelectorAll(".chart-period").forEach((select) => {
    select.addEventListener("change", function () {
      const period = this.value;
      const chartHeader =
        this.closest(".chart-container").querySelector("h3").textContent;
      showNotification(`Updating ${chartHeader} for ${period}`);
    });
  });

  // Refresh metrics button
  const refreshBtn = document.getElementById("refreshMetrics");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", function () {
      this.classList.add("rotating");
      updateAnalyticsData();
      updateTimeAnalysis();

      setTimeout(() => {
        this.classList.remove("rotating");
        showNotification("Analytics data refreshed", "success");
      }, 1000);
    });
  }

  // View details button
  const detailsBtn = document.getElementById("viewDetails");
  if (detailsBtn) {
    detailsBtn.addEventListener("click", function () {
      showNotification("Opening detailed anomaly type analysis");
    });
  }

  // Generate report button
  const reportBtn = document.getElementById("generateReport");
  if (reportBtn) {
    reportBtn.addEventListener("click", function () {
      showNotification("Generating comprehensive analytics report...");

      // Simulate report generation
      setTimeout(() => {
        showNotification("Analytics report generated successfully", "success");

        // Create download link
        const content = "Analytics Report\n================\n\n";
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `analytics_report_${
          new Date().toISOString().split("T")[0]
        }.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 2000);
    });
  }

  // Report download buttons
  document.querySelectorAll(".report-card .btn-outline").forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      const reportType =
        this.closest(".report-card").querySelector("h4").textContent;
      showNotification(`Downloading ${reportType}...`);

      setTimeout(() => {
        showNotification(`${reportType} downloaded successfully`, "success");
      }, 1000);
    });
  });

  // Report card clicks
  document.querySelectorAll(".report-card").forEach((card) => {
    card.addEventListener("click", function () {
      const reportType = this.querySelector("h4").textContent;
      showNotification(`Would generate ${reportType} preview`);
    });
  });
}

function updateAnalyticsForPeriod(period) {
  // Update all analytics data based on selected period
  const periodFactors = {
    today: 0.2,
    week: 0.8,
    month: 1,
    quarter: 1.5,
    year: 3,
  };

  const factor = periodFactors[period] || 1;

  // Update metric cards
  document.querySelectorAll(".metric-card").forEach((card) => {
    const valueElement = card.querySelector(".metric-value");
    const changeElement = card.querySelector(".metric-change");

    if (valueElement && changeElement) {
      const baseValue = parseFloat(valueElement.textContent);
      const newValue = baseValue * factor;

      // Animate to new value
      animateMetric(valueElement, newValue);

      // Update change indicator
      const isPositive = changeElement.classList.contains("positive");
      const newChange = isPositive ? "+2.3%" : "-0.3%";
      changeElement.innerHTML = `<i class="fas fa-arrow-${
        isPositive ? "up" : "down"
      }"></i> ${newChange}`;
    }
  });

  // Update type distribution
  document.querySelectorAll(".type-count").forEach((countElement) => {
    const text = countElement.textContent;
    const match = text.match(/(\d+)\s*events/);
    if (match) {
      const count = parseInt(match[1]);
      const newCount = Math.round(count * factor);
      const percentage = Math.round((newCount / 1247) * 1000) / 10;
      countElement.textContent = `${newCount} events (${percentage}%)`;
    }
  });

  // Update time slots
  document.querySelectorAll(".time-count").forEach((countElement) => {
    const count = parseInt(countElement.textContent);
    const newCount = Math.round(count * factor);

    countElement.style.transform = "scale(1.2)";
    setTimeout(() => {
      countElement.textContent = newCount;
      countElement.style.transform = "scale(1)";
    }, 300);
  });
}

// Add CSS for rotating animation
const style = document.createElement("style");
style.textContent = `
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .rotating {
        animation: rotate 1s linear;
    }
    
    .metric-value.percent::after {
        content: '%';
    }
    
    .metric-value.time::after {
        content: 's';
    }
`;
document.head.appendChild(style);
