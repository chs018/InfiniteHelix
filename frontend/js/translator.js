// Translation functionality
class Translator {
    constructor() {
        this.currentLanguage = 'en';
        this.originalData = null;
        this.translatedData = null;
        this.init();
    }

    init() {
        this.loadLanguages();
    }

    async loadLanguages() {
        try {
            const response = await apiRequest('/translate/languages');
            this.renderLanguageSelector(response.languages);
        } catch (error) {
            console.error('Failed to load languages:', error);
        }
    }

    renderLanguageSelector(languages) {
        const container = document.getElementById('languageSelector');
        if (!container) return;

        // Group languages
        const indian = languages.filter(l => l.category === 'Indian Languages');
        const english = languages.filter(l => l.category === 'Default');
        const others = languages.filter(l => l.category === 'Other Languages');

        let html = `
            <select id="languageSelect" class="language-select">
                <optgroup label="Indian Languages">
                    ${indian.map(l => `
                        <option value="${l.code}" ${l.code === 'en' ? 'selected' : ''}>
                            ${l.native} (${l.name})
                        </option>
                    `).join('')}
                </optgroup>
                <optgroup label="English">
                    ${english.map(l => `
                        <option value="${l.code}" selected>
                            ${l.name}
                        </option>
                    `).join('')}
                </optgroup>
                <optgroup label="Other Languages">
                    ${others.map(l => `
                        <option value="${l.code}">
                            ${l.native} (${l.name})
                        </option>
                    `).join('')}
                </optgroup>
            </select>
            <button id="translateBtn" class="btn btn-primary">
                🌐 Translate
            </button>
            <button id="speakBtn" class="btn btn-secondary" style="display:none;">
                🔊 Speak
            </button>
        `;

        container.innerHTML = html;

        // Event listeners
        document.getElementById('translateBtn').addEventListener('click', () => {
            this.translateCurrentAnalysis();
        });

        document.getElementById('speakBtn').addEventListener('click', () => {
            this.speakTranslation();
        });
    }

    async translateCurrentAnalysis() {
        const analysisId = window.currentAnalysisId;
        if (!analysisId) {
            showToast('No analysis selected', 'error');
            return;
        }

        const selectedLang = document.getElementById('languageSelect').value;
        console.log('Selected language:', selectedLang);
        console.log('Current language:', this.currentLanguage);
        
        if (selectedLang === this.currentLanguage) {
            showToast('Already in selected language', 'info');
            return;
        }

        const translateBtn = document.getElementById('translateBtn');
        translateBtn.disabled = true;
        translateBtn.textContent = '🔄 Translating...';

        try {
            console.log('Sending translation request for analysis:', analysisId);
            const response = await apiRequest(`/translate/${analysisId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    target_language: selectedLang,
                    sections: ['all']
                })
            });

            console.log('Translation response:', response);
            
            this.translatedData = response;
            this.currentLanguage = selectedLang;
            
            // Update UI with translated data
            this.displayTranslatedResults(response);
            
            // Show speak button
            document.getElementById('speakBtn').style.display = 'inline-block';
            
            showToast(`Translated to ${response.language_native}`, 'success');
        } catch (error) {
            console.error('Translation error:', error);
            showToast(`Translation failed: ${error.message || 'Please try again'}`, 'error');
        } finally {
            translateBtn.disabled = false;
            translateBtn.textContent = '🌐 Translate';
        }
    }

    displayTranslatedResults(data) {
        console.log('Displaying translated results:', data);
        
        // Find all insight cards and update them
        const insightCards = document.querySelectorAll('.insight-card');
        if (insightCards.length > 0 && data.insights) {
            insightCards.forEach((card, index) => {
                if (data.insights[index]) {
                    const insight = data.insights[index];
                    
                    // Update title
                    const titleElement = card.querySelector('.insight-title, h4');
                    if (titleElement) {
                        titleElement.textContent = insight.title;
                        // Add RTL if needed
                        if (data.language === 'ar' || data.language === 'ur') {
                            titleElement.classList.add('rtl');
                            titleElement.style.direction = 'rtl';
                            titleElement.style.textAlign = 'right';
                        }
                    }
                    
                    // Update description
                    const descElement = card.querySelector('.insight-description, p');
                    if (descElement) {
                        descElement.textContent = insight.description;
                        // Add RTL if needed
                        if (data.language === 'ar' || data.language === 'ur') {
                            descElement.classList.add('rtl');
                            descElement.style.direction = 'rtl';
                            descElement.style.textAlign = 'right';
                        }
                    }
                }
            });
            console.log(`Updated ${insightCards.length} insight cards`);
        }

        // Find all metric cards and update them
        const metricCards = document.querySelectorAll('.metric-card');
        if (metricCards.length > 0 && data.metrics) {
            metricCards.forEach((card, index) => {
                if (data.metrics[index]) {
                    const metric = data.metrics[index];
                    
                    // Update metric name
                    const nameElement = card.querySelector('.metric-name, h4');
                    if (nameElement) {
                        nameElement.textContent = metric.metric_name;
                        // Add RTL if needed
                        if (data.language === 'ar' || data.language === 'ur') {
                            nameElement.style.direction = 'rtl';
                            nameElement.style.textAlign = 'right';
                        }
                    }
                    
                    // Update notes if present
                    const notesElement = card.querySelector('.metric-notes, [style*="font-style: italic"]');
                    if (notesElement && metric.notes) {
                        notesElement.textContent = metric.notes;
                        // Add RTL if needed
                        if (data.language === 'ar' || data.language === 'ur') {
                            notesElement.style.direction = 'rtl';
                            notesElement.style.textAlign = 'right';
                        }
                    }
                }
            });
            console.log(`Updated ${metricCards.length} metric cards`);
        }

        // Show language badge
        this.showLanguageBadge(data.language_native);
    }

    showLanguageBadge(languageName) {
        let badge = document.getElementById('languageBadge');
        if (!badge) {
            badge = document.createElement('div');
            badge.id = 'languageBadge';
            badge.className = 'language-badge';
            document.querySelector('.results-header')?.appendChild(badge);
        }
        badge.textContent = `🌐 ${languageName}`;
        badge.style.display = 'block';
    }

    getInsightIcon(type) {
        const icons = {
            'summary': '📋',
            'ai_summary': '🤖',
            'risk_assessment': '⚠️',
            'clinical_insight': '🔬',
            'recommendation': '💡',
            'follow_up_plan': '📅',
            'patient_education': '📚',
            'red_flag': '🚩'
        };
        return icons[type] || '📋';
    }

    async speakTranslation() {
        if (!this.translatedData || !this.translatedData.insights.length) {
            showToast('No content to speak', 'error');
            return;
        }

        // Use Web Speech API
        if ('speechSynthesis' in window) {
            const text = this.translatedData.insights
                .map(i => `${i.title}. ${i.description}`)
                .join('. ');
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = this.getVoiceLanguage(this.currentLanguage);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            
            const speakBtn = document.getElementById('speakBtn');
            speakBtn.textContent = '🔊 Speaking...';
            speakBtn.disabled = true;
            
            utterance.onend = () => {
                speakBtn.textContent = '🔊 Speak';
                speakBtn.disabled = false;
            };
            
            utterance.onerror = () => {
                speakBtn.textContent = '🔊 Speak';
                speakBtn.disabled = false;
                showToast('Speech failed', 'error');
            };
            
            speechSynthesis.speak(utterance);
        } else {
            showToast('Speech not supported in this browser', 'error');
        }
    }

    getVoiceLanguage(code) {
        const voiceMap = {
            'hi': 'hi-IN', 'ta': 'ta-IN', 'te': 'te-IN', 'kn': 'kn-IN',
            'ml': 'ml-IN', 'bn': 'bn-IN', 'mr': 'mr-IN', 'gu': 'gu-IN',
            'pa': 'pa-IN', 'ur': 'ur-IN', 'en': 'en-US', 'es': 'es-ES',
            'fr': 'fr-FR', 'de': 'de-DE', 'pt': 'pt-PT', 'ru': 'ru-RU',
            'zh': 'zh-CN', 'ja': 'ja-JP', 'ko': 'ko-KR', 'ar': 'ar-SA'
        };
        return voiceMap[code] || 'en-US';
    }
}

// Initialize on page load
let translator;
document.addEventListener('DOMContentLoaded', () => {
    translator = new Translator();
});
