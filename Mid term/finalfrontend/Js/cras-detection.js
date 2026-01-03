class CrashDetection {
    constructor() {
        this.uploadForm = document.getElementById('crashUploadForm');
        this.videoInput = document.getElementById('crashVideo');
        this.resultsContainer = document.getElementById('crashResults');
        this.init();
    }

    init() {
        this.uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleUpload();
        });
    }

    async handleUpload() {
        const file = this.videoInput.files[0];
        if (!file) return;

        this.showLoading();

        try {
            const formData = new FormData();
            formData.append('video', file);

            const response = await fetch('http://localhost:8000/detect-crash', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            this.showError();
        }
    }

    showLoading() {
        this.resultsContainer.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing video for crash detection...</p>
            </div>
        `;
    }

    showError() {
        this.resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i>
                Error processing video. Please try again.
            </div>
        `;
    }

    displayResults(result) {
        if (result.detected) {
            this.resultsContainer.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-car-crash"></i>
                    <h3>Crash Detected!</h3>
                    <p>Timestamp: ${new Date(result.timestamp).toLocaleString()}</p>
                    <img src="data:image/jpeg;base64,${result.image_base64}" class="crash-image">
                    <button class="btn btn-report">
                        <i class="fas fa-file-alt"></i> Generate Report
                    </button>
                </div>
            `;
        } else {
            this.resultsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-check-circle"></i>
                    No crash detected in the video.
                </div>
            `;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CrashDetection();
});