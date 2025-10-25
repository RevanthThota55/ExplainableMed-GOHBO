/**
 * ExplainableMed-GOHBO - Main JavaScript
 * Handles file upload, prediction, and UI interactions
 */

// Global variables
let selectedFile = null;
let currentPredictionData = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadZone();
    initializeButtons();
});

/**
 * Initialize upload zone with drag & drop functionality
 */
function initializeUploadZone() {
    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

/**
 * Initialize button event listeners
 */
function initializeButtons() {
    // Analyze button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeScan);
    }

    // Change image button
    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', resetUpload);
    }

    // Download report button
    const downloadReportBtn = document.getElementById('downloadReportBtn');
    if (downloadReportBtn) {
        downloadReportBtn.addEventListener('click', downloadReport);
    }

    // Analyze another button
    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    if (analyzeAnotherBtn) {
        analyzeAnotherBtn.addEventListener('click', resetAll);
    }

    // Toggle overlay button
    const toggleOverlayBtn = document.getElementById('toggleOverlay');
    if (toggleOverlayBtn) {
        toggleOverlayBtn.addEventListener('click', toggleHeatmapOverlay);
    }

    // Opacity slider
    const opacitySlider = document.getElementById('opacitySlider');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', updateOpacity);
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

/**
 * Handle file upload and preview
 */
function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, or TIFF)');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        uploadZone.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'inline-flex';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

/**
 * Reset upload UI
 */
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadZone.style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.style.display = 'none';
}

/**
 * Reset all and start over
 */
function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Analyze the uploaded scan
 */
async function analyzeScan() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    // Show loading state
    imagePreview.style.display = 'none';
    analyzeBtn.style.display = 'none';
    loadingState.style.display = 'block';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Store prediction data
            currentPredictionData = data;

            // Hide loading
            loadingState.style.display = 'none';

            // Display results
            displayResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred: ' + error.message);
        loadingState.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'inline-flex';
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    // Scroll to results
    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Update diagnosis
    document.getElementById('diagnosisLabel').textContent = data.prediction;
    document.getElementById('confidenceValue').textContent = data.confidence + '%';

    // Update confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = data.confidence + '%';

    // Update description
    document.getElementById('descriptionText').textContent = data.description;

    // Update images
    document.getElementById('originalImage').src = data.images.original;
    document.getElementById('heatmapImage').src = data.images.overlay;

    // Update probabilities
    updateProbabilities(data.probabilities);

    // Update explanation
    updateExplanation(data.explanation);

    // Set diagnosis color based on result
    const diagnosisLabel = document.getElementById('diagnosisLabel');
    if (data.prediction.toLowerCase().includes('no tumor')) {
        diagnosisLabel.style.color = 'var(--secondary-green)';
    } else {
        diagnosisLabel.style.color = 'var(--accent-red)';
    }
}

/**
 * Update probability bars
 */
function updateProbabilities(probabilities) {
    const container = document.getElementById('probabilitiesContainer');
    container.innerHTML = '';

    for (const [className, probability] of Object.entries(probabilities)) {
        const probabilityPct = (probability * 100).toFixed(2);

        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <div class="probability-label">${className}</div>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${probabilityPct}%">
                    ${probabilityPct}%
                </div>
            </div>
            <div class="probability-value">${probabilityPct}%</div>
        `;

        container.appendChild(item);
    }
}

/**
 * Update explanation section
 */
function updateExplanation(explanation) {
    document.getElementById('diagnosisExplanation').textContent = explanation.diagnosis;
    document.getElementById('confidenceAssessment').textContent = explanation.confidence_assessment;
    document.getElementById('uncertaintyAssessment').textContent = explanation.uncertainty_assessment;

    // Update key findings
    const findingsList = document.getElementById('keyFindingsList');
    findingsList.innerHTML = '';

    explanation.key_findings.forEach(finding => {
        const li = document.createElement('li');
        li.textContent = finding;
        findingsList.appendChild(li);
    });
}

/**
 * Toggle heatmap overlay
 */
function toggleHeatmapOverlay() {
    const heatmapImg = document.getElementById('heatmapImage');

    if (currentPredictionData && currentPredictionData.images) {
        // Toggle between overlay and heatmap only
        if (heatmapImg.src.includes('overlay')) {
            heatmapImg.src = currentPredictionData.images.heatmap;
        } else {
            heatmapImg.src = currentPredictionData.images.overlay;
        }
    }
}

/**
 * Update heatmap opacity
 */
function updateOpacity(event) {
    const opacity = event.target.value;
    document.getElementById('opacityValue').textContent = opacity + '%';

    const heatmapImg = document.getElementById('heatmapImage');
    heatmapImg.style.opacity = opacity / 100;
}

/**
 * Download clinical report
 */
async function downloadReport() {
    if (!currentPredictionData) {
        alert('No prediction data available');
        return;
    }

    try {
        const response = await fetch('/generate_report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(currentPredictionData)
        });

        if (response.ok) {
            // Download the PDF
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `brain_tumor_report_${new Date().getTime()}.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            // Show success message
            showNotification('Report downloaded successfully!', 'success');
        } else {
            throw new Error('Failed to generate report');
        }

    } catch (error) {
        console.error('Error downloading report:', error);
        alert('Error generating report: ' + error.message);
    }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}