/**
 * Advanced ML Features - Frontend JavaScript
 * Fine-tuned BioBERT, Medication NER, Anomaly Detection, Longitudinal Tracking
 */

class AdvancedML {
    constructor() {
        this.currentAnalysisId = null;
        this.init();
    }

    init() {
        console.log('Advanced ML Features initialized');
        this.checkServiceHealth();
    }

    // =================================================================
    // SERVICE HEALTH CHECK
    // =================================================================
    
    async checkServiceHealth() {
        try {
            const response = await apiRequest('/advanced/health-check', {
                method: 'GET'
            });
            
            console.log('Advanced ML Service Status:', response);
            
            // Show feature availability
            this.updateFeatureAvailability(response);
            
        } catch (error) {
            console.error('Advanced ML service check failed:', error);
        }
    }

    updateFeatureAvailability(status) {
        // Update UI based on available features
        const features = {
            'fine_tuned_biobert': status.fine_tuned_biobert,
            'medication_ner': status.medication_ner,
            'anomaly_detector': status.anomaly_detector,
            'longitudinal_tracking': status.longitudinal_tracking
        };

        console.log('Available Advanced Features:', features);
    }

    // =================================================================
    // 1. FINE-TUNED BIOBERT ENTITY EXTRACTION
    // =================================================================
    
    async extractEntitiesAdvanced(analysisId) {
        try {
            console.log('Extracting entities for analysis:', analysisId);
            
            const response = await apiRequest(`/advanced/entity-extraction/${analysisId}`, {
                method: 'POST'
            });
            
            console.log('Entity extraction response:', response);
            
            if (response.entities) {
                this.displayAdvancedEntities(response);
            }
            
            return response;
            
        } catch (error) {
            console.error('Entity extraction failed:', error);
            showToast('Failed to extract entities', 'error');
            throw error;
        }
    }

    displayAdvancedEntities(data) {
        const container = document.getElementById('advancedEntities');
        if (!container) return;

        const { entities, entity_count } = data;
        
        let html = `
            <div class="advanced-feature-card">
                <div class="feature-header">
                    <h3>🧬 Advanced Entity Recognition</h3>
                    <span class="badge badge-success">${entity_count} entities found</span>
                </div>
                <p class="feature-description">Using fine-tuned BioBERT for medical entity extraction</p>
        `;

        // Diseases
        if (entities.diseases && entities.diseases.length > 0) {
            html += `
                <div class="entity-group">
                    <h4>🦠 Diseases & Conditions (${entities.diseases.length})</h4>
                    <div class="entity-list">
                        ${entities.diseases.map(e => `
                            <div class="entity-badge">
                                <span class="entity-text">${e.text}</span>
                                <span class="entity-score">${(e.score * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Chemicals/Drugs
        if (entities.chemicals && entities.chemicals.length > 0) {
            html += `
                <div class="entity-group">
                    <h4>💊 Chemicals & Drugs (${entities.chemicals.length})</h4>
                    <div class="entity-list">
                        ${entities.chemicals.map(e => `
                            <div class="entity-badge">
                                <span class="entity-text">${e.text}</span>
                                <span class="entity-score">${(e.score * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Genes
        if (entities.genes && entities.genes.length > 0) {
            html += `
                <div class="entity-group">
                    <h4>🧬 Genes (${entities.genes.length})</h4>
                    <div class="entity-list">
                        ${entities.genes.map(e => `
                            <div class="entity-badge">
                                <span class="entity-text">${e.text}</span>
                                <span class="entity-score">${(e.score * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        // Proteins
        if (entities.proteins && entities.proteins.length > 0) {
            html += `
                <div class="entity-group">
                    <h4>🧪 Proteins (${entities.proteins.length})</h4>
                    <div class="entity-list">
                        ${entities.proteins.map(e => `
                            <div class="entity-badge">
                                <span class="entity-text">${e.text}</span>
                                <span class="entity-score">${(e.score * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        html += `</div>`;
        container.innerHTML = html;
        container.style.display = 'block';
    }

    // =================================================================
    // 2. CUSTOM MEDICATION NER
    // =================================================================
    
    async extractMedications(analysisId) {
        try {
            console.log('Extracting medications for analysis:', analysisId);
            
            const response = await apiRequest(`/advanced/medication-extraction/${analysisId}`, {
                method: 'POST'
            });
            
            console.log('Medication extraction response:', response);
            
            if (response.medications) {
                this.displayMedications(response);
            }
            
            return response;
            
        } catch (error) {
            console.error('Medication extraction failed:', error);
            showToast('Failed to extract medications', 'error');
            throw error;
        }
    }

    displayMedications(data) {
        const container = document.getElementById('medications');
        if (!container) return;

        const { medications, medication_count } = data;
        
        if (medication_count === 0) {
            container.innerHTML = `
                <div class="advanced-feature-card">
                    <h3>💊 Medications</h3>
                    <p>No medications detected in this report.</p>
                </div>
            `;
            container.style.display = 'block';
            return;
        }

        let html = `
            <div class="advanced-feature-card">
                <div class="feature-header">
                    <h3>💊 Medication Analysis</h3>
                    <span class="badge badge-info">${medication_count} medications</span>
                </div>
                <p class="feature-description">Custom NER model for medication extraction</p>
                
                <div class="medication-list">
        `;

        medications.forEach(med => {
            const confidenceClass = med.confidence >= 0.75 ? 'high' : med.confidence >= 0.5 ? 'medium' : 'low';
            
            html += `
                <div class="medication-card">
                    <div class="medication-header">
                        <span class="medication-name">${med.drug_name}</span>
                        <span class="confidence-badge ${confidenceClass}">${(med.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div class="medication-details">
                        <div class="detail-item">
                            <span class="detail-label">Dosage:</span>
                            <span class="detail-value">${med.dosage}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Frequency:</span>
                            <span class="detail-value">${med.frequency}</span>
                        </div>
                    </div>
                    <div class="medication-context">${med.context}</div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        container.style.display = 'block';
    }

    // =================================================================
    // 3. LAB RESULT ANOMALY DETECTION
    // =================================================================
    
    async detectAnomalies(analysisId) {
        try {
            console.log('Detecting anomalies for analysis:', analysisId);
            
            const response = await apiRequest(`/advanced/anomaly-detection/${analysisId}`, {
                method: 'POST'
            });
            
            console.log('Anomaly detection response:', response);
            
            if (response.anomalies) {
                this.displayAnomalies(response);
            }
            
            return response;
            
        } catch (error) {
            console.error('Anomaly detection failed:', error);
            showToast('Failed to detect anomalies', 'error');
            throw error;
        }
    }

    displayAnomalies(data) {
        const container = document.getElementById('anomalies');
        if (!container) return;

        const { anomalies, anomaly_count, total_metrics } = data;
        
        let html = `
            <div class="advanced-feature-card anomaly-card">
                <div class="feature-header">
                    <h3>⚠️ Anomaly Detection</h3>
                    <span class="badge ${anomaly_count > 0 ? 'badge-warning' : 'badge-success'}">
                        ${anomaly_count} / ${total_metrics} anomalous
                    </span>
                </div>
                <p class="feature-description">Isolation Forest algorithm for outlier detection</p>
        `;

        if (anomaly_count === 0) {
            html += `
                <div class="no-anomalies">
                    <p>✅ All lab results within expected patterns!</p>
                </div>
            `;
        } else {
            html += `<div class="anomaly-list">`;
            
            anomalies.forEach(anomaly => {
                const severityIcon = anomaly.anomaly_severity === 'high' ? '🔴' : '🟡';
                
                html += `
                    <div class="anomaly-item ${anomaly.anomaly_severity}">
                        <div class="anomaly-header">
                            <span class="anomaly-icon">${severityIcon}</span>
                            <span class="anomaly-metric">${anomaly.metric_name}</span>
                            <span class="anomaly-value">${anomaly.metric_value} ${anomaly.unit || ''}</span>
                        </div>
                        <div class="anomaly-reason">${anomaly.anomaly_reason}</div>
                        <div class="anomaly-score">
                            Anomaly Score: <strong>${anomaly.anomaly_score.toFixed(3)}</strong>
                        </div>
                    </div>
                `;
            });
            
            html += `</div>`;
        }

        html += `</div>`;
        container.innerHTML = html;
        container.style.display = 'block';
    }

    // =================================================================
    // 4. LONGITUDINAL HEALTH TRACKING
    // =================================================================
    
    async analyzeLongitudinal() {
        try {
            console.log('Analyzing longitudinal health trends...');
            
            const response = await apiRequest('/advanced/longitudinal-analysis', {
                method: 'GET'
            });
            
            console.log('Longitudinal analysis response:', response);
            
            if (response.trends) {
                this.displayLongitudinalAnalysis(response);
            }
            
            return response;
            
        } catch (error) {
            console.error('Longitudinal analysis failed:', error);
            showToast('Failed to analyze trends', 'error');
            throw error;
        }
    }

    displayLongitudinalAnalysis(data) {
        const container = document.getElementById('longitudinalAnalysis');
        if (!container) return;

        const { trends, predictions, risk_changes, insights, data_points, tracking_period_days } = data;
        
        let html = `
            <div class="advanced-feature-card longitudinal-card">
                <div class="feature-header">
                    <h3>📈 Longitudinal Health Tracking</h3>
                    <span class="badge badge-primary">${data_points} checkups over ${tracking_period_days} days</span>
                </div>
                <p class="feature-description">Time-series analysis of your health metrics</p>
        `;

        // Insights
        if (insights && insights.length > 0) {
            html += `
                <div class="insights-section">
                    <h4>💡 Key Insights</h4>
                    <ul class="insights-list">
                        ${insights.map(insight => `<li>${insight}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Trends
        if (trends && trends.length > 0) {
            html += `
                <div class="trends-section">
                    <h4>📊 Metric Trends</h4>
                    <div class="trends-grid">
            `;
            
            trends.forEach(trend => {
                const directionIcon = trend.direction === 'increasing' ? '📈' : 
                                     trend.direction === 'decreasing' ? '📉' : '➡️';
                const directionClass = trend.direction;
                
                html += `
                    <div class="trend-card ${directionClass}">
                        <div class="trend-header">
                            <span class="trend-icon">${directionIcon}</span>
                            <span class="trend-name">${trend.metric_name}</span>
                        </div>
                        <div class="trend-values">
                            <div class="value-item">
                                <span class="value-label">From:</span>
                                <span class="value-number">${trend.first_value}</span>
                            </div>
                            <div class="value-item">
                                <span class="value-label">To:</span>
                                <span class="value-number">${trend.last_value}</span>
                            </div>
                        </div>
                        <div class="trend-change">
                            Change: <strong>${trend.change > 0 ? '+' : ''}${trend.change}</strong>
                            (${trend.change_percent > 0 ? '+' : ''}${trend.change_percent}%)
                        </div>
                        <div class="trend-period">
                            Over ${trend.time_span_days} days (${trend.data_points} points)
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }

        // Predictions
        if (predictions && predictions.length > 0) {
            html += `
                <div class="predictions-section">
                    <h4>🔮 Future Predictions (30 days)</h4>
                    <div class="predictions-list">
            `;
            
            predictions.forEach(pred => {
                html += `
                    <div class="prediction-card">
                        <div class="prediction-metric">${pred.metric_name}</div>
                        <div class="prediction-values">
                            <span class="current">Current: ${pred.current_value}</span>
                            <span class="arrow">→</span>
                            <span class="predicted">Predicted: ${pred.predicted_value}</span>
                        </div>
                        <div class="prediction-confidence">
                            Confidence: <span class="badge badge-${pred.confidence}">${pred.confidence}</span>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }

        // Risk Changes
        if (risk_changes && risk_changes.length > 0) {
            html += `
                <div class="risk-changes-section">
                    <h4>⚠️ Risk Level Changes</h4>
                    <div class="risk-changes-list">
            `;
            
            risk_changes.forEach(risk => {
                const severityClass = risk.severity === 'high' ? 'danger' : 'warning';
                
                html += `
                    <div class="risk-change-item ${severityClass}">
                        <span class="risk-icon">⚠️</span>
                        <span class="risk-message">${risk.message}</span>
                        <span class="risk-badge badge-${severityClass}">${risk.severity}</span>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }

        html += `</div>`;
        container.innerHTML = html;
        container.style.display = 'block';
    }

    // =================================================================
    // COMBINED ADVANCED ANALYSIS
    // =================================================================
    
    async runFullAdvancedAnalysis(analysisId) {
        this.currentAnalysisId = analysisId;
        
        try {
            // Run all advanced features in parallel
            const results = await Promise.allSettled([
                this.extractEntitiesAdvanced(analysisId),
                this.extractMedications(analysisId),
                this.detectAnomalies(analysisId),
                this.analyzeLongitudinal()
            ]);

            console.log('Advanced analysis complete:', results);
            
            // Count successful analyses
            const successful = results.filter(r => r.status === 'fulfilled').length;
            
            showToast(
                `Advanced analysis complete! ${successful}/4 features executed successfully.`,
                'success'
            );
            
            return results;
            
        } catch (error) {
            console.error('Error running full advanced analysis:', error);
            showToast('Some advanced features failed', 'warning');
        }
    }
}

// Initialize global instance
const advancedML = new AdvancedML();

// Expose to window
window.advancedML = advancedML;
