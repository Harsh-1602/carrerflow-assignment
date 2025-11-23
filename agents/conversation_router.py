"""
Conversation Router - Intelligently routes user queries to appropriate agents using LLM
"""
import json
from typing import Dict, Optional, List
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os


class AgentType(Enum):
    """Types of available agents"""
    COMPANY_RESEARCH = "company_research"
    JOB_MATCHING = "job_matching"
    SECTION_ENHANCEMENT = "section_enhancement"
    GENERAL = "general"


class RouteDecision(BaseModel):
    """Structured output for routing decision"""
    agent_type: str = Field(description="The type of agent to route to: company_research, job_matching, section_enhancement, or general")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")
    extracted_entities: Dict = Field(description="Extracted entities like company_name, section_name, etc.")
    requires_followup: bool = Field(description="Whether the query needs follow-up questions")


class ConversationRouter:
    """
    Routes user queries to the appropriate specialized agent using LLM-based classification
    """
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        """Initialize router with LLM"""
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = PydanticOutputParser(pydantic_object=RouteDecision)
        self.routing_prompt = self._build_routing_prompt()
    
    def _build_routing_prompt(self) -> ChatPromptTemplate:
        """Build the LLM prompt for routing decisions"""
        
        system_message = """You are an intelligent conversation router for a resume optimization system. 
Your job is to analyze user queries and determine which specialized agent should handle them.

Available Agents:

1. COMPANY_RESEARCH Agent:
   - Handles: Optimizing resumes for specific companies
   - Examples:
     * "Optimize my resume for Google"
     * "Tailor my resume to match Amazon's culture"
     * "Make my resume fit Apple's values"
     * "Adapt my resume for Microsoft"
   
2. JOB_MATCHING Agent:
   - Handles: Matching resumes to job descriptions, analyzing fit
   - Examples:
     * "How well does my resume match this job description?"
     * "Analyze this job posting and improve my resume"
     * "Update my resume to match these requirements"
     * "Check my ATS compatibility for this role"
   
3. SECTION_ENHANCEMENT Agent:
   - Handles: Improving specific resume sections
   - Examples:
     * "Improve my experience section"
     * "Make my skills more impactful"
     * "Add quantification to my achievements"
     * "Enhance my summary statement"
     * "Strengthen my education section"

4. GENERAL Agent:
   - Handles: General questions, greetings, unclear requests
   - Examples:
     * "Hello"
     * "What can you do?"
     * "Help me with my resume"
     * "I need advice"

Instructions:
- Analyze the user's query carefully
- Choose the MOST appropriate agent based on the intent
- If the query is ambiguous or could fit multiple categories, choose the most specific match
- If the query is a general question or greeting, use GENERAL agent
- Extract any relevant entities (company names, section names, etc.)
- Indicate if follow-up questions are needed for clarity

{format_instructions}"""

        human_message = """User Query: {query}

Previous Context (if any): {context}

Analyze this query and provide your routing decision:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def route_query(self, user_query: str, context: Optional[Dict] = None) -> Dict:
        """
        Determine which agent should handle the query using LLM
        
        Args:
            user_query: User's input text
            context: Optional conversation context
            
        Returns:
            Routing decision with agent type and confidence
        """
        try:
            # Prepare context string
            context_str = json.dumps(context) if context else "No previous context"
            
            # Get format instructions and format the prompt
            format_instructions = self.parser.get_format_instructions()
            
            # Format the messages directly
            messages = self.routing_prompt.format_messages(
                format_instructions=format_instructions,
                query=user_query,
                context=context_str
            )
            
            response = self.llm.invoke(messages)
            decision = self.parser.parse(response.content)
            
            # Convert to AgentType enum
            agent_type_map = {
                "company_research": AgentType.COMPANY_RESEARCH,
                "job_matching": AgentType.JOB_MATCHING,
                "section_enhancement": AgentType.SECTION_ENHANCEMENT,
                "general": AgentType.GENERAL
            }
            
            agent_type = agent_type_map.get(
                decision.agent_type.lower(),
                AgentType.GENERAL
            )
            
            return {
                "agent_type": agent_type,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "entities": decision.extracted_entities,
                "requires_followup": decision.requires_followup
            }
            
        except Exception as e:
            print(f"Error in LLM routing: {e}")
            # Fallback to general agent
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {str(e)}",
                "entities": {},
                "requires_followup": True
            }
            return {
                "agent_type": AgentType.GENERAL,
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {str(e)}",
                "entities": {},
                "requires_followup": True
            }
    
    def suggest_next_steps(self, current_agent: AgentType, 
                          conversation_history: List[Dict]) -> List[str]:
        """
        Suggest logical next steps based on current context
        
        Args:
            current_agent: Currently active agent
            conversation_history: Previous conversation turns
            
        Returns:
            List of suggested next actions
        """
        suggestions = {
            AgentType.COMPANY_RESEARCH: [
                "Would you like me to analyze how well your resume matches this company?",
                "Should I enhance specific sections to better align with this company?",
                "Would you like me to compare your resume against a specific job description?"
            ],
            AgentType.JOB_MATCHING: [
                "Should I enhance the sections that need the most improvement?",
                "Would you like me to research the company culture for better alignment?",
                "Shall I work on improving your match score?"
            ],
            AgentType.SECTION_ENHANCEMENT: [
                "Would you like me to work on other sections as well?",
                "Should I optimize the entire resume for a specific company or role?",
                "Would you like me to check how well this matches a job description?"
            ],
            AgentType.GENERAL: [
                "What would you like to improve in your resume?",
                "Are you targeting a specific company or role?",
                "Which section would you like to enhance?"
            ]
        }
        
        return suggestions.get(current_agent, suggestions[AgentType.GENERAL])
    
    def parse_complex_query(self, query: str) -> List[Dict]:
        """
        Parse complex queries that may require multiple agents using LLM
        
        Args:
            query: User query
            
        Returns:
            List of sub-tasks with agent assignments
        """
        try:
            # Use LLM to detect if query has multiple intents
            detection_prompt = f"""Analyze this user query and determine if it contains multiple distinct requests that should be handled separately:

Query: "{query}"

If the query contains multiple requests (connected by "and", "also", "then", etc.), break it down into separate tasks.
If it's a single request, return it as one task.

Return a JSON list of tasks in this format:
[
  {{"task_number": 1, "description": "task description", "intent": "company_research|job_matching|section_enhancement|general"}},
  ...
]

Response:"""
            
            response = self.llm.invoke(detection_prompt)
            
            # Try to parse JSON response
            try:
                tasks_data = json.loads(response.content.strip().strip('```json').strip('```'))
                
                tasks = []
                for i, task_data in enumerate(tasks_data):
                    # Map intent to agent type
                    intent = task_data.get("intent", "general").lower()
                    agent_type_map = {
                        "company_research": AgentType.COMPANY_RESEARCH,
                        "job_matching": AgentType.JOB_MATCHING,
                        "section_enhancement": AgentType.SECTION_ENHANCEMENT,
                        "general": AgentType.GENERAL
                    }
                    
                    tasks.append({
                        "task_id": i + 1,
                        "description": task_data.get("description", query),
                        "agent_type": agent_type_map.get(intent, AgentType.GENERAL),
                        "priority": i + 1
                    })
                
                return tasks if tasks else self._fallback_single_task(query)
                
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as single task
                return self._fallback_single_task(query)
                
        except Exception as e:
            print(f"Error in complex query parsing: {e}")
            return self._fallback_single_task(query)
    
    def _fallback_single_task(self, query: str) -> List[Dict]:
        """Fallback to single task routing"""
        routing = self.route_query(query)
        return [{
            "task_id": 1,
            "description": query,
            "agent_type": routing["agent_type"],
            "priority": 1
        }]


# Global router instance
conversation_router = ConversationRouter()
