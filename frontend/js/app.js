/**
 * Main application logic for GNN Vertex Degree Predictor
 */

const API_BASE_URL = 'http://localhost:5555';

/**
 * Generate a random graph and populate input fields
 */
async function generateRandomGraph() {
    const generateBtn = document.getElementById('generateBtn');
    const graphInput = document.getElementById('graphInput');
    const vertexInput = document.getElementById('vertexInput');
    
    // Disable button while generating
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'GET',
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate graph');
        }

        // Populate the input fields
        graphInput.value = data.graph;
        vertexInput.value = data.vertex;

    } catch (error) {
        showError('Error generating graph: ' + error.message);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Random Graph';
    }
}

/**
 * Analyze the graph and display results
 */
async function analyzeGraph() {
    const graphInput = document.getElementById('graphInput').value.trim();
    const vertexInput = document.getElementById('vertexInput').value.trim();
    const errorMessage = document.getElementById('errorMessage');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const analyzeBtn = document.getElementById('analyzeBtn');

    // Reset UI
    hideError();
    hideResults();

    // Validate input
    if (!graphInput) {
        showError('Please enter a graph (edge list)');
        return;
    }

    if (vertexInput === '') {
        showError('Please enter a target vertex');
        return;
    }

    // Show loading
    showLoading();
    analyzeBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                graph: graphInput,
                vertex: parseInt(vertexInput)
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'An error occurred');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        hideLoading();
        analyzeBtn.disabled = false;
    }
}

/**
 * Display the analysis results
 */
function displayResults(data) {
    document.getElementById('trueDegree').textContent = data.true_degree;
    document.getElementById('predictedDegree').textContent = data.predicted_degree;
    
    const graphImage = document.getElementById('graphImage');
    const placeholder = document.getElementById('placeholder');
    
    graphImage.src = 'data:image/png;base64,' + data.graph_image;
    graphImage.classList.add('show');
    placeholder.classList.add('hide');
}

/**
 * Show error message
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

/**
 * Hide error message
 */
function hideError() {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.classList.remove('show');
    errorMessage.textContent = '';
}

/**
 * Show loading spinner
 */
function showLoading() {
    const loading = document.getElementById('loading');
    loading.classList.add('show');
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    const loading = document.getElementById('loading');
    loading.classList.remove('show');
}

/**
 * Show results section
 */
function showResults() {
    // Results are always visible in new layout
}

/**
 * Hide results section
 */
function hideResults() {
    const graphImage = document.getElementById('graphImage');
    const placeholder = document.getElementById('placeholder');
    
    graphImage.classList.remove('show');
    placeholder.classList.remove('hide');
    
    document.getElementById('trueDegree').textContent = '-';
    document.getElementById('predictedDegree').textContent = '-';
}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    // Allow Enter key in vertex input
    document.getElementById('vertexInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeGraph();
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeEventListeners);
