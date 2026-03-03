// Results display functionality

async function loadResults(analysisId) {
    try {
        window.currentAnalysisId = analysisId;
        
        // Show loading state
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '<div class="spinner"></div>';
        
        // Load analysis data
        const [analysis, metrics, insights] = await Promise.all([
            apiRequest(`/results/${analysisId}`),
            apiRequest(`/results/${analysisId}/metrics`),
            apiRequest(`/results/${analysisId}/insights`)
        ]);

        displayResults(analysis, metrics, insights);
        
        // Start voice chat session for this analysis
        if (window.voiceChatbot) {
            window.voiceChatbot.startSession(analysisId);
        }
        
        // Switch to results section (but don't trigger loadResults again)
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById('results').classList.add('active');
        
        // Update nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('data-section') === 'results') {
                link.classList.add('active');
            }
        });
        
    } catch (error) {
        console.error('Failed to load results:', error);
        document.getElementById('resultsContainer').innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <p class="empty-state-text">Failed to load results. Please try again.</p>
            </div>
        `;
    }
}

function displayResults(analysis, metrics, insights) {
    const container = document.getElementById('resultsContainer');
    
    container.innerHTML = `
        ${displaySummary(analysis)}
        ${displayMetrics(metrics)}
        ${displayInsights(insights)}
    `;
}

function displaySummary(analysis) {
    return `
        <div class="results-summary">
            <div class="summary-header">
                <h3 class="summary-title">Analysis Summary</h3>
                <div class="results-actions">
                    <button class="btn btn-secondary" onclick="exportResults()">Export</button>
                    <button class="btn btn-secondary" onclick="copyResultsToClipboard()">Copy</button>
                </div>
            </div>
            
            <div class="summary-info">
                <div class="info-item">
                    <div class="info-label">Analysis Date</div>
                    <div class="info-value">${formatDate(analysis.analysis_date)}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Status</div>
                    <div class="info-value">${analysis.status || 'Completed'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">OCR Confidence</div>
                    <div class="info-value">${analysis.ocr_confidence ? (analysis.ocr_confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Processing Time</div>
                    <div class="info-value">${analysis.processing_time ? analysis.processing_time.toFixed(2) + 's' : 'N/A'}</div>
                </div>
            </div>

            ${analysis.extracted_text ? `
                <div style="margin-top: 1.5rem;">
                    <h4 style="margin-bottom: 0.5rem;">Extracted Text (Preview)</h4>
                    <div style="background: var(--bg-secondary); padding: 1rem; border-radius: var(--radius-md); max-height: 200px; overflow-y: auto; font-size: 0.875rem; color: var(--text-secondary);">
                        ${escapeHtml(analysis.extracted_text.substring(0, 500))}${analysis.extracted_text.length > 500 ? '...' : ''}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function displayMetrics(metrics) {
    if (!metrics || metrics.length === 0) {
        return '<p style="color: var(--text-secondary); text-align: center;">No medical metrics found.</p>';
    }

    return `
        <div style="margin-bottom: 2rem;">
            <h3 style="font-size: 1.5rem; margin-bottom: 1.5rem;">Medical Metrics</h3>
            <div class="metrics-grid">
                ${metrics.map(metric => createMetricCard(metric)).join('')}
            </div>
        </div>
    `;
}

function createMetricCard(metric) {
    const statusClass = (metric.status || 'normal').toLowerCase();
    const statusDisplay = metric.status || 'Normal';
    
    return `
        <div class="metric-card">
            <div class="metric-header">
                <h4 class="metric-name">${escapeHtml(metric.metric_name)}</h4>
                <span class="status-badge ${statusClass}">${statusDisplay}</span>
            </div>
            <div class="metric-value-display">
                ${escapeHtml(metric.metric_value)}${metric.metric_unit ? ' ' + escapeHtml(metric.metric_unit) : ''}
            </div>
            ${metric.reference_range ? `
                <div class="metric-reference">
                    Reference Range: ${escapeHtml(metric.reference_range)}
                </div>
            ` : ''}
            ${metric.category ? `
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    Category: ${escapeHtml(metric.category)}
                </div>
            ` : ''}
            ${metric.notes ? `
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem; font-style: italic;">
                    ${escapeHtml(metric.notes)}
                </div>
            ` : ''}
        </div>
    `;
}

function displayInsights(insights) {
    if (!insights || insights.length === 0) {
        return '<p style="color: var(--text-secondary); text-align: center;">No health insights generated.</p>';
    }

    // Sort by priority
    const sortedInsights = [...insights].sort((a, b) => (b.priority || 0) - (a.priority || 0));

    return `
        <div>
            <h3 style="font-size: 1.5rem; margin-bottom: 1.5rem;">Health Insights</h3>
            <div class="insights-list">
                ${sortedInsights.map(insight => createInsightCard(insight)).join('')}
            </div>
        </div>
    `;
}

function createInsightCard(insight) {
    const severityClass = (insight.severity || 'info').toLowerCase();
    
    return `
        <div class="insight-card ${severityClass}">
            <div class="insight-header">
                <h4 class="insight-title">${escapeHtml(insight.title)}</h4>
                <span class="insight-type">${escapeHtml(insight.insight_type)}</span>
            </div>
            <p class="insight-description">${escapeHtml(insight.description)}</p>
            ${insight.is_actionable ? `
                <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--primary-color); font-weight: 600;">
                    ✓ Actionable Item
                </div>
            ` : ''}
        </div>
    `;
}

function exportResults() {
    showToast('Export functionality coming soon!', 'info');
    // TODO: Implement PDF export
}

function copyResultsToClipboard() {
    const container = document.getElementById('resultsContainer');
    const text = container.innerText;
    
    navigator.clipboard.writeText(text).then(() => {
        showToast('Results copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy to clipboard', 'error');
    });
}

// Export functions to global scope
window.loadResults = loadResults;
window.exportResults = exportResults;
window.copyResultsToClipboard = copyResultsToClipboard;
