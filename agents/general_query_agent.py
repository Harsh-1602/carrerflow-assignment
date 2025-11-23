"""
General Query Agent
Handles general questions about resume, career advice, and context-based queries using LLM
"""
from openai import OpenAI
from config import settings
from typing import Dict, Optional


class GeneralQueryAgent:
    """Agent that handles general queries about resume and career using LLM"""
    
    def __init__(self):
        """Initialize the General Query Agent"""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    def answer_query(self, query: str, resume_text: str, 
                     conversation_context: Optional[str] = None) -> Dict:
        """
        Answer general queries about resume or career advice
        
        Args:
            query: User's question
            resume_text: Current resume content for context
            conversation_context: Optional conversation history
            
        Returns:
            Dict with answer and metadata
        """
        system_prompt = """You are an expert career advisor and resume consultant. 
You help users with questions about their resume, career guidance, and professional development.

Your responsibilities:
1. Answer questions about the user's resume content
2. Provide career advice and guidance
3. Explain resume optimization concepts
4. Help with general career-related questions
5. Be conversational and helpful

Guidelines:
- Be specific and reference actual content from the resume when relevant
- Provide actionable advice
- Keep responses concise but informative (2-4 paragraphs max)
- Be encouraging and professional
- If you don't have enough context, ask clarifying questions
- Don't make assumptions about information not in the resume"""

        # Build context
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation context if available
        if conversation_context:
            messages.append({
                "role": "system", 
                "content": f"Previous conversation context:\n{conversation_context}"
            })
        
        # Add resume context
        messages.append({
            "role": "system",
            "content": f"User's Current Resume:\n{resume_text[:3000]}..."  # Limit to avoid token limits
        })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "status": "success",
                "answer": answer,
                "agent_type": "general_query"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "agent_type": "general_query"
            }
    
    def answer_about_resume(self, query: str, resume_text: str) -> str:
        """
        Answer specific questions about resume content
        
        Args:
            query: Question about the resume
            resume_text: Resume content
            
        Returns:
            Answer string
        """
        result = self.answer_query(query, resume_text)
        return result.get("answer", "I couldn't generate an answer.")
    
    def provide_career_advice(self, query: str, resume_text: str) -> str:
        """
        Provide career advice based on resume and query
        
        Args:
            query: Career-related question
            resume_text: Resume content for context
            
        Returns:
            Career advice
        """
        system_prompt = """You are an experienced career coach. Provide practical, 
actionable career advice based on the user's background and their question. 
Consider their experience level, skills, and career trajectory visible in their resume."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": f"User's Resume:\n{resume_text[:3000]}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I couldn't provide career advice at this time: {str(e)}"
    
    def explain_concept(self, concept: str, resume_context: Optional[str] = None) -> str:
        """
        Explain resume or career-related concepts
        
        Args:
            concept: Concept to explain (e.g., "ATS optimization", "action verbs")
            resume_context: Optional resume context to make explanation relevant
            
        Returns:
            Explanation string
        """
        system_prompt = """You are a resume and career education expert. 
Explain concepts clearly with examples. If resume context is provided, 
make the explanation relevant to that specific resume."""
        
        context_text = f"\n\nUser's Resume Context:\n{resume_context[:2000]}" if resume_context else ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Explain: {concept}{context_text}"}
                ],
                temperature=0.6,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I couldn't explain that concept: {str(e)}"
    
    def handle_greeting(self, greeting: str, has_resume: bool = False) -> str:
        """
        Handle greetings and initial interactions
        
        Args:
            greeting: User's greeting message
            has_resume: Whether user has uploaded a resume
            
        Returns:
            Friendly response
        """
        if has_resume:
            return """Hello! I'm your AI resume optimization assistant. I've loaded your resume and I'm ready to help!

I can help you with:
• **Company Optimization** - Tailor your resume for specific companies
• **Job Matching** - Optimize for job descriptions and improve match scores
• **Section Enhancement** - Improve specific sections with better impact
• **Questions & Advice** - Answer questions about your resume or provide career guidance

What would you like to work on today?"""
        else:
            return """Hello! I'm your AI resume optimization assistant.

To get started, please upload your resume (PDF or DOCX format). Once uploaded, I can help you:
• Optimize for specific companies
• Match against job descriptions
• Enhance specific sections
• Answer questions about resume best practices

Upload your resume to begin!"""
    
    def suggest_next_steps(self, resume_text: str, previous_actions: Optional[list] = None) -> str:
        """
        Suggest next steps based on current resume state
        
        Args:
            resume_text: Current resume content
            previous_actions: List of actions already taken
            
        Returns:
            Suggestions for next steps
        """
        actions_context = ""
        if previous_actions:
            actions_context = f"\n\nActions already taken: {', '.join(previous_actions)}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a resume optimization strategist. 
Based on the resume and what's been done so far, suggest 2-3 specific next steps 
to further improve the resume. Be concise and actionable."""},
                    {"role": "user", "content": f"Resume:\n{resume_text[:2000]}{actions_context}\n\nWhat should be the next steps?"}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return "Continue optimizing your resume by targeting specific companies or enhancing key sections."


# Global instance
general_query_agent = GeneralQueryAgent()
