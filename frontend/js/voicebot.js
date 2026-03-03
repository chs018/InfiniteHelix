// Voice Chatbot functionality
class VoiceChatbot {
    constructor() {
        this.sessionId = null;
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.messages = [];
        this.init();
    }

    init() {
        // Initialize Speech Recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.handleVoiceInput(transcript);
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopListening();
                showToast('Voice recognition error. Try again.', 'error');
            };

            this.recognition.onend = () => {
                this.stopListening();
            };
        }
    }

    async startSession(analysisId) {
        try {
            const response = await apiRequest('/chat/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    analysis_id: analysisId
                })
            });

            this.sessionId = response.session_id;
            this.messages = [];
            
            // Display chat interface
            this.renderChatInterface();
            
            // Add welcome message
            this.addMessage('assistant', response.message);
            
            // Show suggested questions
            if (response.suggestions && response.suggestions.length > 0) {
                this.showSuggestions(response.suggestions);
            }
            
            showToast('Voice chat ready!', 'success');
        } catch (error) {
            console.error('Failed to start chat:', error);
            showToast('Failed to start voice chat', 'error');
        }
    }

    renderChatInterface() {
        const container = document.getElementById('voiceChatContainer');
        if (!container) {
            // Create container
            const newContainer = document.createElement('div');
            newContainer.id = 'voiceChatContainer';
            newContainer.className = 'voice-chat-container';
            newContainer.innerHTML = `
                <div class="chat-header">
                    <h3>🎤 Medical Assistant</h3>
                    <button id="closeChatBtn" class="btn-close">✕</button>
                </div>
                <div id="chatMessages" class="chat-messages"></div>
                <div id="chatSuggestions" class="chat-suggestions"></div>
                <div class="chat-controls">
                    <button id="voiceBtn" class="btn-voice" title="Click to speak">
                        🎤 Speak
                    </button>
                    <input type="text" id="chatInput" placeholder="Type or speak your question..." />
                    <button id="sendBtn" class="btn-send">Send</button>
                </div>
            `;
            document.body.appendChild(newContainer);

            // Event listeners
            document.getElementById('voiceBtn').addEventListener('click', () => {
                this.toggleVoiceInput();
            });

            document.getElementById('sendBtn').addEventListener('click', () => {
                const input = document.getElementById('chatInput');
                if (input.value.trim()) {
                    this.sendMessage(input.value.trim());
                    input.value = '';
                }
            });

            document.getElementById('chatInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const input = e.target;
                    if (input.value.trim()) {
                        this.sendMessage(input.value.trim());
                        input.value = '';
                    }
                }
            });

            document.getElementById('closeChatBtn').addEventListener('click', () => {
                this.closeChat();
            });
        } else {
            container.style.display = 'flex';
        }
    }

    toggleVoiceInput() {
        if (!this.recognition) {
            showToast('Voice input not supported in this browser', 'error');
            return;
        }

        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    }

    startListening() {
        if (!this.recognition) return;

        try {
            this.recognition.start();
            this.isListening = true;
            
            const voiceBtn = document.getElementById('voiceBtn');
            voiceBtn.textContent = '🔴 Listening...';
            voiceBtn.classList.add('listening');
            
            showToast('Listening... Speak now', 'info');
        } catch (error) {
            console.error('Failed to start recognition:', error);
        }
    }

    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
        
        this.isListening = false;
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            voiceBtn.textContent = '🎤 Speak';
            voiceBtn.classList.remove('listening');
        }
    }

    handleVoiceInput(transcript) {
        console.log('Voice input:', transcript);
        this.sendMessage(transcript);
    }

    async sendMessage(message) {
        if (!this.sessionId) {
            showToast('Chat session not started', 'error');
            return;
        }

        // Add user message
        this.addMessage('user', message);

        try {
            const response = await apiRequest('/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    message: message
                })
            });

            // Add assistant response
            this.addMessage('assistant', response.response);
            
            // Speak response
            this.speak(response.response);
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        }
    }

    addMessage(role, content) {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}`;
        
        const timestamp = new Date().toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${content}</div>
                <div class="message-time">${timestamp}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        this.messages.push({ role, content, timestamp });
    }

    showSuggestions(suggestions) {
        const suggestionsContainer = document.getElementById('chatSuggestions');
        if (!suggestionsContainer) return;

        let html = '<div class="suggestions-title">Suggested questions:</div>';
        suggestions.forEach(suggestion => {
            html += `
                <button class="suggestion-btn" onclick="voiceChatbot.sendMessage('${suggestion.replace(/'/g, "\\'")}')">
                    ${suggestion}
                </button>
            `;
        });

        suggestionsContainer.innerHTML = html;
    }

    speak(text) {
        if (this.isSpeaking) {
            this.synthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 1;
        
        utterance.onstart = () => {
            this.isSpeaking = true;
        };
        
        utterance.onend = () => {
            this.isSpeaking = false;
        };
        
        this.synthesis.speak(utterance);
    }

    stopSpeaking() {
        if (this.isSpeaking) {
            this.synthesis.cancel();
            this.isSpeaking = false;
        }
    }

    closeChat() {
        const container = document.getElementById('voiceChatContainer');
        if (container) {
            container.style.display = 'none';
        }
        this.stopListening();
        this.stopSpeaking();
    }

    openChat() {
        const container = document.getElementById('voiceChatContainer');
        if (container) {
            container.style.display = 'flex';
        } else {
            // Start new session with current analysis
            if (window.currentAnalysisId) {
                this.startSession(window.currentAnalysisId);
            }
        }
    }
}

// Initialize on page load
let voiceChatbot;
document.addEventListener('DOMContentLoaded', () => {
    voiceChatbot = new VoiceChatbot();
});
