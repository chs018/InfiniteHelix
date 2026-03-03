// API Configuration
const API_BASE_URL = 'http://localhost:8000/api/v1';

// Global state on window object
window.currentAnalysisId = null;
window.currentFileId = null;
window.authToken = null;
window.userData = null;

// Check authentication
function checkAuth() {
    const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data') || sessionStorage.getItem('user_data');
    
    if (!token) {
        // Redirect to login page
        window.location.href = 'auth.html';
        return false;
    }
    
    window.authToken = token;
    window.userData = userData ? JSON.parse(userData) : null;
    
    // Display user info
    if (window.userData) {
        displayUserInfo(window.userData);
    }
    
    return true;
}

// Display user information
function displayUserInfo(user) {
    const userInfoHtml = `
        <div class="user-info">
            <span class="user-name">👤 ${user.name || user.email}</span>
            <button onclick="logout()" class="btn-logout">Logout</button>
        </div>
    `;
    
    // Add to header if not exists
    const header = document.querySelector('.header');
    if (header && !document.querySelector('.user-info')) {
        header.insertAdjacentHTML('beforeend', userInfoHtml);
    }
}

// Logout function
function logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    sessionStorage.removeItem('auth_token');
    sessionStorage.removeItem('user_data');
    window.location.href = 'auth.html';
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication first
    if (!checkAuth()) {
        return;
    }
    
    initNavigation();
    loadHistory();
});

// Navigation
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            showSection(section);
        });
    });
}

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });

    // Show target section
    const targetSection = document.getElementById(sectionName);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === sectionName) {
            link.classList.add('active');
        }
    });

    // Load data for specific sections
    if (sectionName === 'history') {
        loadHistory();
    } else if (sectionName === 'results' && window.currentAnalysisId) {
        loadResults(window.currentAnalysisId);
    }
}

// API Helper
async function apiRequest(endpoint, options = {}) {
    try {
        // Get auth token
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        
        // Add authorization header if token exists
        const headers = {
            ...options.headers,
        };
        
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            ...options,
            headers: headers,
        });

        // Handle unauthorized responses
        if (response.status === 401) {
            // Token expired or invalid, redirect to login
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user_data');
            sessionStorage.removeItem('auth_token');
            sessionStorage.removeItem('user_data');
            window.location.href = 'auth.html';
            return;
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showToast(error.message || 'An error occurred', 'error');
        throw error;
    }
}

// Toast Notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    
    toastMessage.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// History
async function loadHistory(page = 1, limit = 10) {
    try {
        const data = await apiRequest(`/history?skip=${(page - 1) * limit}&limit=${limit}`);
        displayHistory(data);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

function displayHistory(data) {
    const container = document.getElementById('historyContainer');
    
    if (!data.items || data.items.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <p class="empty-state-text">No history available yet.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="history-list">
            ${data.items.map(item => `
                <div class="history-item" onclick="loadResults('${item.analysis_id}')">
                    <div class="history-item-header">
                        <h3 class="history-item-title">${item.report_type || 'Medical Report'}</h3>
                        <span class="history-item-date">${formatDate(item.created_date)}</span>
                    </div>
                    <p class="history-item-details">Status: ${item.overall_status || 'Completed'}</p>
                </div>
            `).join('')}
        </div>
    `;

    // Show pagination if needed
    if (data.total > data.limit) {
        displayPagination(data);
    }
}

function displayPagination(data) {
    const container = document.getElementById('historyPagination');
    const totalPages = Math.ceil(data.total / data.limit);
    const currentPage = Math.floor(data.skip / data.limit) + 1;

    let paginationHTML = '';
    for (let i = 1; i <= totalPages; i++) {
        paginationHTML += `
            <button 
                class="pagination-btn ${i === currentPage ? 'active' : ''}"
                onclick="loadHistory(${i})"
            >
                ${i}
            </button>
        `;
    }

    container.innerHTML = paginationHTML;
    container.style.display = 'flex';
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions to global scope
window.showSection = showSection;
window.loadHistory = loadHistory;
window.showToast = showToast;
window.apiRequest = apiRequest;
window.formatFileSize = formatFileSize;
window.formatDate = formatDate;
window.escapeHtml = escapeHtml;
