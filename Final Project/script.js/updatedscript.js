// Main dashboard functionality
class SmartZoneDashboard {
    constructor() {
        this.currentSection = 'home';
        this.init();
    }

    init() {
        this.loadSection('home');
        this.setupEventListeners();
        this.setupSearch();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.currentTarget.getAttribute('data-section');
                this.loadSection(section);
                
                // Update active state
                document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
                e.currentTarget.classList.add('active');
            });
        });

        // Sidebar toggle
        document.querySelector('.sidebar-toggle').addEventListener('click', () => {
            document.querySelector('.sidebar').classList.toggle('collapsed');
        });
    }

    async loadSection(section) {
        this.currentSection = section;
        
        try {
            const response = await fetch(`sections/${section}.html`);
            const html = await response.text();
            
            document.getElementById('content-area').innerHTML = html;
            
            // Initialize section-specific functionality
            this.initSection(section);
            
        } catch (error) {
            console.error('Error loading section:', error);
            document.getElementById('content-area').innerHTML = '<div class="error">Error loading section</div>';
        }
    }

    initSection(section) {
        switch(section) {
            case 'monitoring':
                this.initMonitoring();
                break;
            case 'upload':
                this.initUpload();
                break;
            case 'analytics':
                this.initAnalytics();
                break;
            case 'about':
                this.initAbout();
                break;
        }
    }

    initMonitoring() {
        const toggleCameraBtn = document.getElementById('toggleCamera');
        const captureBtn = document.getElementById('captureBtn');
        const cctvFeed = document.getElementById('cctvFeed');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const statusDot = document.getElementById('statusDot');
        const cameraStatus = document.getElementById('cameraStatus');
        const peopleCount = document.getElementById('peopleCount');

        let isCameraOn = false;

        toggleCameraBtn.addEventListener('click', function() {
            isCameraOn = !isCameraOn;
            if (isCameraOn) {
                // Simulate starting camera
                cctvFeed.style.display = 'block';
                cameraPlaceholder.style.display = 'none';
                toggleCameraBtn.innerHTML = '<i class="fas fa-power-off"></i> Stop Camera';
                captureBtn.disabled = false;
                statusDot.className = 'status-dot status-safe';
                cameraStatus.textContent = 'Active';
                peopleCount.textContent = '3'; // Simulate people count
            } else {
                cctvFeed.style.display = 'none';
                cameraPlaceholder.style.display = 'flex';
                toggleCameraBtn.innerHTML = '<i class="fas fa-power-off"></i> Start Camera';
                captureBtn.disabled = true;
                statusDot.className = 'status-dot';
                cameraStatus.textContent = 'Inactive';
                peopleCount.textContent = '0';
            }
        });

        captureBtn.addEventListener('click', function() {
            // Simulate capture
            alert('Snapshot captured!');
        });
    }

    initUpload() {
        const dropZone = document.getElementById('dropZone');
        const videoUpload = document.getElementById('videoUpload');
        const uploadBtn = document.getElementById('uploadBtn');
        const uploadProgress = document.querySelector('.upload-progress');
        const videoPreview = document.getElementById('videoPreview');

        uploadBtn.addEventListener('click', function() {
            videoUpload.click();
        });

        videoUpload.addEventListener('change', function(e) {
            handleFileSelect(e);
        });

        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', function() {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) {
                handleFileSelect({ target: { files: files } });
            }
        });

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && file.type.startsWith('video/')) {
                // Show preview
                videoPreview.src = URL.createObjectURL(file);
                videoPreview.style.display = 'block';
                // Simulate upload progress
                uploadProgress.style.display = 'block';
                let progress = 0;
                const interval = setInterval(() => {
                    progress += 10;
                    document.querySelector('.progress').style.width = progress + '%';
                    document.querySelector('.progress-text').textContent = progress + '% uploaded';
                    if (progress >= 100) {
                        clearInterval(interval);
                    }
                }, 200);
            }
        }
    }

    initAnalytics() {
        this.loadMapDependencies().then(() => {
            if (!window.mapInitialized) {
                this.initMap();
                window.mapInitialized = true;
            }
        });
    }

    loadMapDependencies() {
        return new Promise((resolve) => {
            if (window.L) return resolve();

            const script1 = document.createElement('script');
            script1.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
            script1.onload = () => {
                const script2 = document.createElement('script');
                script2.src = 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js';
                script2.onload = () => {
                    const script3 = document.createElement('script');
                    script3.src = 'https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js';
                    script3.onload = resolve;
                    document.body.appendChild(script3);
                };
                document.body.appendChild(script2);
            };
            document.body.appendChild(script1);
        });
    }

    initMap() {
        const map = L.map('map').setView([27.7172, 85.3240], 13); // Kathmandu coordinates

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        // Add some sample markers
        L.marker([27.7172, 85.3240]).addTo(map)
            .bindPopup('Kathmandu')
            .openPopup();

        // You can add more map features here
    }

    initAbout() {
        const videoThumbnail = document.getElementById('videoThumbnail');
        const videoModal = document.getElementById('videoModal');
        const tutorialVideo = document.getElementById('tutorialVideo');
        const closeBtn = document.querySelector('.close-btn');

        if (videoThumbnail) {
            videoThumbnail.addEventListener('click', function() {
                videoModal.style.display = 'block';
                tutorialVideo.play();
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', function() {
                videoModal.style.display = 'none';
                tutorialVideo.pause();
            });
        }
    }

    setupSearch() {
        const searchInput = document.getElementById('dashboardSearch');
        const searchResults = document.getElementById('searchResults');
        const closeResults = document.querySelector('.close-results');

        searchInput.addEventListener('focus', () => {
            searchResults.style.display = 'block';
        });

        closeResults.addEventListener('click', () => {
            searchResults.style.display = 'none';
        });

        searchInput.addEventListener('input', (e) => {
            // Implement search functionality here
            console.log('Searching for:', e.target.value);
        });
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SmartZoneDashboard();
});

// API functions
async function analyzeCrash(data) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    return await response.json();
}