"""
Voice chatbot service with medical context
"""
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class VoiceChatService:
    """
    Voice chatbot service for medical Q&A
    """
    
    def __init__(self):
        api_key = os.getenv('GROQ_API_KEY', '')
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Voice chat will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            logger.info("Voice Chat Service initialized successfully")
        
        self.model = "llama-3.3-70b-versatile"
        self.conversations = {}  # In-memory storage for conversations
    
    def is_available(self) -> bool:
        """Check if voice chat service is available"""
        return self.client is not None
    
    def create_session(self, user_id: int, analysis_id: str) -> str:
        """Create a new chat session"""
        session_id = f"{user_id}_{analysis_id}_{datetime.utcnow().timestamp()}"
        self.conversations[session_id] = {
            'user_id': user_id,
            'analysis_id': analysis_id,
            'messages': [],
            'created_at': datetime.utcnow()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get chat session"""
        return self.conversations.get(session_id)
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        medical_context: Dict[str, Any]
    ) -> str:
        """
        Process user message and generate response
        """
        if not self.is_available():
            return "Voice chat service is currently unavailable."
        
        session = self.get_session(session_id)
        if not session:
            return "Session not found. Please start a new conversation."
        
        try:
            # Build context from medical data
            context_parts = ["# Medical Report Context\n"]
            
            if medical_context.get('summary'):
                context_parts.append(f"Summary: {medical_context['summary']}\n")
            
            if medical_context.get('metrics'):
                context_parts.append("\nMedical Metrics:")
                for metric in medical_context['metrics'][:5]:  # Top 5 metrics
                    context_parts.append(
                        f"- {metric['metric_name']}: {metric['metric_value']} "
                        f"{metric['metric_unit']} ({metric['status']})"
                    )
            
            if medical_context.get('insights'):
                context_parts.append("\n\nHealth Insights:")
                for insight in medical_context['insights'][:3]:  # Top 3 insights
                    context_parts.append(f"- {insight['title']}: {insight['description']}")
            
            context_text = "\n".join(context_parts)
            
            # Build conversation history
            history = []
            for msg in session['messages'][-5:]:  # Last 5 messages for context
                history.append(f"User: {msg['user']}")
                history.append(f"Assistant: {msg['assistant']}")
            
            history_text = "\n".join(history) if history else "No previous conversation."
            
            # Create prompt
            prompt = f"""You are a helpful medical assistant helping a patient understand their medical report. 

{context_text}

Previous Conversation:
{history_text}

Current User Question: {user_message}

Instructions:
1. Answer based ONLY on the provided medical data
2. Be clear, concise, and compassionate
3. Explain medical terms in simple language
4. If asked about something not in the data, say you don't have that information
5. Always remind them to consult their doctor for medical decisions
6. Keep responses under 150 words for voice delivery

Your Response:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            assistant_message = response.choices[0].message.content.strip()
            
            # Store in conversation history
            session['messages'].append({
                'user': user_message,
                'assistant': assistant_message,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"Chat response generated for session {session_id}")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I'm having trouble processing your question. Please try again."
    
    async def get_suggestions(self, medical_context: Dict[str, Any]) -> List[str]:
        """
        Generate suggested questions based on medical data
        """
        if not self.is_available():
            return [
                "What do my test results mean?",
                "Are my values normal?",
                "What should I do next?"
            ]
        
        try:
            context_summary = ""
            if medical_context.get('metrics'):
                metrics_names = [m['metric_name'] for m in medical_context['metrics'][:3]]
                context_summary = f"Metrics: {', '.join(metrics_names)}"
            
            prompt = f"""Based on this medical report data:
{context_summary}

Generate 3 natural questions a patient might ask. Return ONLY the questions, one per line, no numbering.

Examples:
What does my blood pressure reading mean?
Are my glucose levels normal?
What lifestyle changes do you recommend?

Your 3 questions:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=150
            )
            
            suggestions = response.choices[0].message.content.strip().split('\n')
            return [s.strip() for s in suggestions if s.strip()][:3]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [
                "What do my test results mean?",
                "Are my values normal?",
                "What should I do next?"
            ]
    
    def clear_old_sessions(self, hours: int = 24):
        """Clear sessions older than specified hours"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        old_sessions = [
            sid for sid, session in self.conversations.items()
            if session['created_at'].timestamp() < cutoff
        ]
        for sid in old_sessions:
            del self.conversations[sid]
        logger.info(f"Cleared {len(old_sessions)} old chat sessions")


# Global instance
voice_chat_service = VoiceChatService()
