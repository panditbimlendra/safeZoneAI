// ==================== MAIN APPLICATION CONTROLLER ====================
class DashboardApp {
  constructor() {
    this.initializeApp();
  }

  initializeApp() {
    this.setupNavigation();
    this.setupUIComponents();
    this.setupTimeDisplay();
    this.setupSidebar();
    this.setupFileUpload();
    this.setupSearch();
    this.setupVideoPlayer();
    this.setupCameraControls();
    this.initializeMap();
  }

  // ==================== NAVIGATION SYSTEM ====================
  setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        this.handleNavigation(link);
      });
    });

    // Initialize first section
    const defaultSection = document.querySelector('.dashboard-section.active');
    if (defaultSection) {
      this.activateSection(defaultSection.id);
    }
  }

  handleNavigation(link) {
    // Update active nav link
    document.querySelectorAll('.nav-links a').forEach(a => {
      a.classList.remove('active');
    });
    link.classList.add('active');

    // Get target section
    const sectionId = link.getAttribute('data-section');
    this.activateSection(sectionId);
  }

  activateSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.dashboard-section').forEach(section => {
      section.classList.remove('active');
    });

    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
      targetSection.classList.add('active');
      this.loadSectionResources(sectionId);
    } else {
      console.error(`Section ${sectionId} not found`);
    }
  }

  loadSectionResources(sectionId) {
    switch(sectionId) {
      case 'analytics':
        this.initializeMap();
        break;
      case 'monitoring':
        this.setupCameraControls();
        break;
      case 'upload':
        this.setupFileUpload();
        break;
    }
  }

  // ==================== UI COMPONENTS ====================
  setupUIComponents() {
    // Option cards
    document.querySelectorAll('.option-card').forEach(card => {
      card.addEventListener('click', () => {
        document.querySelectorAll('.option-card').forEach(c => {
          c.classList.remove('active');
        });
        card.classList.add('active');
      });
    });

    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => {
          b.classList.remove('active');
        });
        btn.classList.add('active');
      });
    });
  }

  // ==================== TIME DISPLAY ====================
  setupTimeDisplay() {
    this.updateKathmanduTime();
  }

  updateKathmanduTime() {
    const now = new Date();
    const offset = 5.75 * 60 * 60 * 1000; // UTC+5:45
    const kathmanduTime = new Date(now.getTime() + offset);
    
    // Format date and time
    const dateStr = kathmanduTime.toISOString().split('T')[0];
    const timeStr = kathmanduTime.toTimeString().substring(0, 8);
    
    // Update DOM
    const dateElement = document.getElementById('kathmandu-date') || 
                       document.querySelector('.date-display');
    if (dateElement) dateElement.textContent = dateStr;
    
    // Update every second
    setTimeout(() => this.updateKathmanduTime(), 1000);
  }

  // ==================== SIDEBAR CONTROLS ====================
  setupSidebar() {
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (sidebarToggle && sidebar) {
      sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
      });

      // Load saved state
      if (localStorage.getItem('sidebarCollapsed') === 'true') {
        sidebar.classList.add('collapsed');
      }
    }
  }

  // ==================== FILE UPLOAD SYSTEM ====================
  setupFileUpload() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;

    const handlers = {
      preventDefaults: (e) => {
        e.preventDefault();
        e.stopPropagation();
      },
      highlight: () => dropZone.classList.add('highlight'),
      unhighlight: () => dropZone.classList.remove('highlight'),
      handleDrop: (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles({ target: { files } });
      },
      handleFiles: (e) => {
        const files = e.target.files;
        if (files.length) {
          const file = files[0];
          if (!file.type.match('video.*')) {
            alert('Please select a video file');
            return;
          }
          this.previewVideo(file);
          this.uploadVideo(file);
        }
      }
    };

    // Add event listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, handlers.preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, handlers.highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, handlers.unhighlight, false);
    });

    dropZone.addEventListener('drop', handlers.handleDrop);
    document.getElementById('videoUpload')?.addEventListener('change', handlers.handleFiles);
    document.getElementById('uploadBtn')?.addEventListener('click', () => {
      document.getElementById('videoUpload')?.click();
    });
  }

  previewVideo(file) {
    const videoPreview = document.getElementById('videoPreview');
    if (videoPreview) {
      videoPreview.src = URL.createObjectURL(file);
      videoPreview.style.display = 'block';
    }
  }

  uploadVideo(file) {
    const progressContainer = document.querySelector('.upload-progress');
    const progressBar = document.querySelector('.progress');
    const progressText = document.querySelector('.progress-text');

    if (!progressContainer || !progressBar || !progressText) return;

    progressContainer.style.display = 'block';
    
    const formData = new FormData();
    formData.append('video', file);
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);
    
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percentComplete = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = `${percentComplete}%`;
        progressText.textContent = `${percentComplete}% uploaded`;
      }
    };
    
    xhr.onload = () => {
      progressContainer.style.display = 'none';
      if (xhr.status === 200) {
        alert('Video uploaded successfully!');
      } else {
        alert(`Upload failed: ${xhr.statusText}`);
      }
    };
    
    xhr.send(formData);
  }

  // ==================== SEARCH SYSTEM ====================
  setupSearch() {
    const searchInput = document.querySelector('.search-bar input');
    const searchIcon = document.querySelector('.search-bar .fa-search');

    if (!searchInput || !searchIcon) return;

    const performSearch = (query) => {
      if (!query.trim()) return;
      
      const results = this.searchAllSections(query);
      this.displaySearchResults(results, query);
    };

    searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') performSearch(searchInput.value);
    });

    searchIcon.addEventListener('click', () => performSearch(searchInput.value));
  }

  searchAllSections(query) {
    const results = [];
    const lowerQuery = query.toLowerCase();

    // Search detection logs
    document.querySelectorAll('.detection-item').forEach(item => {
      if (item.textContent.toLowerCase().includes(lowerQuery)) {
        results.push({
          type: 'detection',
          element: item,
          text: item.textContent.trim()
        });
      }
    });

    // Search alert logs
    document.querySelectorAll('.alert-table tbody tr').forEach(row => {
      if (row.textContent.toLowerCase().includes(lowerQuery)) {
        results.push({
          type: 'alert',
          element: row,
          text: row.textContent.trim()
        });
      }
    });

    return results;
  }

  displaySearchResults(results, query) {
    document.querySelectorAll('.search-highlight').forEach(el => {
      el.classList.remove('search-highlight');
    });

    if (results.length === 0) {
      alert(`No results found for: ${query}`);
      return;
    }

    results.forEach(result => {
      result.element.classList.add('search-highlight');
    });

    results[0].element.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  }

  // ==================== VIDEO PLAYER ====================
  setupVideoPlayer() {
    const thumbnail = document.getElementById('videoThumbnail');
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('tutorialVideo');
    const closeBtn = document.querySelector('.close-btn');
    const fullscreenBtn = document.querySelector('.fullscreen-btn');

    if (!thumbnail || !modal || !video) return;

    const videoControls = {
      openModal: () => {
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        video.play().catch(error => {
          console.error('Video playback failed:', error);
          alert('Video playback failed. Please try again.');
        });
      },
      closeModal: () => {
        modal.style.display = 'none';
        document.body.style.overflow = '';
        video.pause();
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
      },
      toggleFullscreen: () => {
        if (!document.fullscreenElement) {
          video.requestFullscreen().catch(console.error);
        } else {
          document.exitFullscreen();
        }
      }
    };

    thumbnail.addEventListener('click', videoControls.openModal);
    closeBtn?.addEventListener('click', videoControls.closeModal);
    fullscreenBtn?.addEventListener('click', videoControls.toggleFullscreen);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) videoControls.closeModal();
    });

    document.addEventListener('keydown', (e) => {
      if (modal.style.display === 'block') {
        if (e.key === 'Escape') videoControls.closeModal();
        if (e.key === 'f') videoControls.toggleFullscreen();
      }
    });
  }

  // ==================== CAMERA CONTROLS ====================
  setupCameraControls() {
    const video = document.getElementById('cctvFeed');
    const placeholder = document.getElementById('cameraPlaceholder');
    const toggleBtn = document.getElementById('toggleCamera');
    const captureBtn = document.getElementById('captureBtn');
    const statusText = document.getElementById('cameraStatus');
    const statusDot = document.getElementById('statusDot');
    const peopleCount = document.getElementById('peopleCount');

    if (!video || !toggleBtn) return;

    let stream = null;
    let detectionInterval = null;

    const camera = {
      start: async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
          });
          
          video.srcObject = stream;
          video.style.display = 'block';
          placeholder.style.display = 'none';
          this.updateCameraUI(true);
          this.startDetection();
        } catch (err) {
          console.error('Camera error:', err);
          statusText.textContent = 'Error';
          statusDot.className = 'status-dot';
          alert(`Could not access camera: ${err.message}`);
        }
      },
      stop: () => {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
          stream = null;
        }
        video.srcObject = null;
        video.style.display = 'none';
        placeholder.style.display = 'flex';
        this.stopDetection();
        this.updateCameraUI(false);
      },
      capture: () => {
        if (!stream) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.font = '16px Arial';
        ctx.fillStyle = 'white';
        ctx.fillText(new Date().toLocaleString(), 10, 20);
        
        const link = document.createElement('a');
        link.download = `cctv-${new Date().toISOString().slice(0, 10)}.jpg`;
        link.href = canvas.toDataURL('image/jpeg');
        link.click();
      },
      updateUI: (isActive) => {
        if (isActive) {
          statusText.textContent = 'Active';
          statusDot.className = 'status-dot status-active';
          toggleBtn.innerHTML = '<i class="fas fa-power-off"></i> Stop Camera';
          captureBtn.disabled = false;
        } else {
          statusText.textContent = 'Inactive';
          statusDot.className = 'status-dot';
          toggleBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Camera';
          captureBtn.disabled = true;
          peopleCount.textContent = '0';
        }
      },
      startDetection: () => {
        detectionInterval = setInterval(() => {
          const randomCount = Math.floor(Math.random() * 5);
          peopleCount.textContent = randomCount;
          statusDot.className = randomCount > 0 ? 
            'status-dot status-warning' : 'status-dot status-active';
        }, 3000);
      },
      stopDetection: () => {
        if (detectionInterval) {
          clearInterval(detectionInterval);
          detectionInterval = null;
        }
      }
    };

    // Assign methods to instance
    this.updateCameraUI = camera.updateUI;
    this.startDetection = camera.startDetection;
    this.stopDetection = camera.stopDetection;

    // Add event listeners
    toggleBtn.addEventListener('click', async () => {
      if (stream) {
        await camera.stop();
      } else {
        await camera.start();
      }
    });

    captureBtn?.addEventListener('click', camera.capture);
    window.addEventListener('beforeunload', () => camera.stop());
  }

  // ==================== MAP SYSTEM ====================
  initializeMap() {
    if (window.mapInitialized || !document.getElementById('map')) return;

    const incidents = [
      { lat: 27.7172, lng: 85.3240, intensity: 5, type: "fire", title: "Kathmandu Fire" },
      // ... other incidents
    ];

    const map = L.map('map').setView([27.7172, 85.3240], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    // Layer setup
    const heatLayer = L.layerGroup();
    const markerLayer = L.layerGroup();
    const clusterLayer = L.markerClusterGroup();

    // Process incidents
    incidents.forEach(incident => {
      const marker = L.marker([incident.lat, incident.lng], {
        icon: this.getMapIcon(incident.type)
      }).bindPopup(`<b>${incident.title}</b><br>Type: ${incident.type}`);
      
      markerLayer.addLayer(marker);
      clusterLayer.addLayer(marker.clone());
    });

    // Heatmap
    const heatData = incidents.map(incident => [incident.lat, incident.lng, incident.intensity]);
    if (heatData.length > 0) {
      L.heatLayer(heatData, { radius: 25 }).addTo(heatLayer);
    }

    // Layer control
    const baseLayers = {
      "Street Map": L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'),
      "Satellite": L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}')
    };

    const overlays = {
      "Heatmap": heatLayer,
      "Markers": markerLayer,
      "Clusters": clusterLayer
    };

    L.control.layers(baseLayers, overlays).addTo(map);
    heatLayer.addTo(map);

    // Layer switcher
    document.getElementById('mapLayer')?.addEventListener('change', function() {
      map.eachLayer(layer => {
        if (layer !== baseLayers["Street Map"] && layer !== baseLayers["Satellite"]) {
          map.removeLayer(layer);
        }
      });
      
      switch(this.value) {
        case 'heatmap': heatLayer.addTo(map); break;
        case 'markers': markerLayer.addTo(map); break;
        case 'clusters': clusterLayer.addTo(map); break;
      }
    });

    window.mapInitialized = true;
  }

  getMapIcon(type) {
    const iconUrl = {
      'fire': 'https://cdn-icons-png.flaticon.com/512/2936/2936886.png',
      'crowd': 'https://cdn-icons-png.flaticon.com/512/484/484167.png',
      'fall': 'https://cdn-icons-png.flaticon.com/512/3764/3764342.png',
      'crash': 'https://cdn-icons-png.flaticon.com/512/836/836576.png'
    }[type] || 'https://cdn-icons-png.flaticon.com/512/447/447031.png';
    
    return L.icon({
      iconUrl: iconUrl,
      iconSize: [32, 32],
      popupAnchor: [0, -15]
    });
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new DashboardApp();


  
});

