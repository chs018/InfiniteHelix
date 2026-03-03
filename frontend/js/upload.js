// Upload functionality
let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOMContentLoaded - Initializing upload...');
    initUpload();
});

function initUpload() {
    console.log('🔧 initUpload called');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const removeFileBtn = document.getElementById('removeFile');
    const chooseFileBtn = document.getElementById('chooseFileBtn');

    console.log('📦 Elements found:', {
        dropZone: !!dropZone,
        fileInput: !!fileInput,
        uploadBtn: !!uploadBtn,
        removeFileBtn: !!removeFileBtn,
        chooseFileBtn: !!chooseFileBtn
    });

    if (!dropZone || !fileInput || !uploadBtn || !removeFileBtn || !chooseFileBtn) {
        console.error('❌ Upload elements not found!');
        return;
    }

    console.log('✅ All elements found, setting up listeners...');

    // Choose file button click - trigger file input
    chooseFileBtn.addEventListener('click', function(e) {
        console.log('🖱️ Choose file button clicked!');
        e.preventDefault();
        e.stopPropagation();
        fileInput.value = ''; // Reset input
        console.log('📂 Triggering file input click...');
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', function(e) {
        console.log('📁 File input changed!', e.target.files);
        const file = e.target.files[0];
        if (file) {
            console.log('✅ File selected:', file.name, file.size, file.type);
            validateAndSetFile(file);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            validateAndSetFile(file);
        }
    });

    // Upload button
    uploadBtn.addEventListener('click', () => {
        if (selectedFile) {
            uploadAndAnalyze();
        }
    });

    // Remove file button
    removeFileBtn.addEventListener('click', () => {
        clearFile();
    });
}

function validateAndSetFile(file) {
    // Check file size (10MB max)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size exceeds 10MB limit', 'error');
        return;
    }

    // Check file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'text/plain'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload PDF, PNG, JPG, or TXT files.', 'error');
        return;
    }

    selectedFile = file;
    displayFileInfo(file);
}

function displayFileInfo(file) {
    document.getElementById('dropZone').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'block';
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('uploadBtn').disabled = false;
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('progressContainer').style.display = 'none';
}

async function uploadAndAnalyze() {
    if (!selectedFile) return;

    const uploadBtn = document.getElementById('uploadBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    try {
        // Disable upload button
        uploadBtn.disabled = true;
        uploadBtn.querySelector('#uploadBtnText').textContent = 'Uploading...';
        
        // Show progress
        progressContainer.style.display = 'block';
        progressBar.style.width = '30%';
        progressText.textContent = 'Uploading file...';

        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Get auth token
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        const headers = {};
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            headers: headers,
            body: formData,
        });

        if (!uploadResponse.ok) {
            throw new Error('Upload failed');
        }

        const uploadData = await uploadResponse.json();
        window.currentFileId = uploadData.file_id;
        console.log('✅ File uploaded! File ID:', window.currentFileId);

        progressBar.style.width = '60%';
        progressText.textContent = 'Starting analysis...';

        // Start analysis
        console.log('📤 Starting analysis for file:', window.currentFileId);
        const analyzeResponse = await apiRequest(`/analyze/${window.currentFileId}`, {
            method: 'POST',
        });

        window.currentAnalysisId = analyzeResponse.id;  // ✅ Fixed: use 'id' not 'analysis_id'
        console.log('✅ Analysis started! Analysis ID:', window.currentAnalysisId);
        console.log('📊 Full analyze response:', analyzeResponse);

        progressBar.style.width = '80%';
        progressText.textContent = 'Processing report...';

        // Poll for results
        console.log('🔄 Starting to poll for analysis:', window.currentAnalysisId);
        await pollAnalysisResults(window.currentAnalysisId);

        progressBar.style.width = '100%';
        progressText.textContent = 'Analysis complete!';

        showToast('Analysis completed successfully!', 'success');

        // Clear form and load results
        setTimeout(() => {
            clearFile();
            loadResults(window.currentAnalysisId);  // Load results directly, don't use showSection
        }, 1000);

    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message || 'Upload failed. Please try again.', 'error');
        uploadBtn.disabled = false;
        uploadBtn.querySelector('#uploadBtnText').textContent = 'Upload and Analyze';
        progressContainer.style.display = 'none';
    }
}

async function pollAnalysisResults(analysisId, maxAttempts = 30) {
    console.log('🔍 Polling for analysis ID:', analysisId);
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        try {
            console.log(`📡 Poll attempt ${attempts + 1}/${maxAttempts} for ID: ${analysisId}`);
            const response = await apiRequest(`/analyze/${analysisId}/status`);
            
            if (response.status === 'completed') {
                return response;
            } else if (response.status === 'failed') {
                throw new Error(response.error_message || 'Analysis failed');
            }
            
            // Wait 2 seconds before next poll
            await new Promise(resolve => setTimeout(resolve, 2000));
            attempts++;
            
        } catch (error) {
            if (attempts >= maxAttempts - 1) {
                throw error;
            }
            await new Promise(resolve => setTimeout(resolve, 2000));
            attempts++;
        }
    }
    
    throw new Error('Analysis timeout. Please check results later.');
}
