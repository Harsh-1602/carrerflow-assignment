"""
Agent orchestration - coordinates multiple agents and manages conversation flow
"""
from typing import Dict, List, Optional
from agents.company_research_agent import CompanyResearchAgent
from agents.job_matching_agent import JobMatchingAgent
from agents.section_enhancement_agent import SectionEnhancementAgent
from agents.general_query_agent import GeneralQueryAgent
from agents.conversation_router import conversation_router, AgentType
from services.firebase_service import firebase_service
from services.vector_store import vector_store
from tools.resume_parser import ResumeParser
from utils.output_cleaner import output_cleaner


class ResumeOptimizationOrchestrator:
    """
    Main orchestrator that coordinates agents and manages the optimization workflow
    """
    
    def __init__(self):
        """Initialize orchestrator with all agents"""
        self.company_agent = CompanyResearchAgent()
        self.job_agent = JobMatchingAgent()
        self.section_agent = SectionEnhancementAgent()
        self.general_agent = GeneralQueryAgent()
        self.router = conversation_router
        
        # Current state
        self.current_session = None
        self.current_resume = None
        self.conversation_history = []
    
    def start_session(self, resume_path: str, user_id: str = None) -> Dict:
        """
        Start a new optimization session
        
        Args:
            resume_path: Path to resume file
            user_id: Optional user identifier
            
        Returns:
            Session information
        """
        # Parse resume
        try:
            parsed_resume = ResumeParser.parse_resume(resume_path)
            
            # Create session
            session_id = firebase_service.create_session(user_id)
            
            # Store resume
            version_id = firebase_service.save_resume_version(
                session_id,
                parsed_resume["raw_text"],
                "Original",
                {"file_name": parsed_resume["file_name"]}
            )
            
            # Add to vector store
            vector_store.add_resume_to_index(
                session_id,
                parsed_resume["raw_text"],
                {"version_id": version_id}
            )
            
            # Update state
            self.current_session = session_id
            self.current_resume = parsed_resume
            self.conversation_history = []
            
            # Add initial message
            firebase_service.add_message(
                session_id,
                "system",
                "Resume loaded successfully. How can I help you optimize it?",
                {"resume_file": parsed_resume["file_name"]}
            )
            
            return {
                "session_id": session_id,
                "resume_info": {
                    "file_name": parsed_resume["file_name"],
                    "word_count": parsed_resume["word_count"],
                    "sections": list(parsed_resume["sections"].keys())
                },
                "status": "success"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user query and route to appropriate agent
        Uses RAG pattern: first searches vector store, falls back to full document if needed
        
        Args:
            query: User's question or request
            context: Optional additional context including conversation_history
            
        Returns:
            Response with result and metadata
        """
        if not self.current_session:
            return {
                "status": "error",
                "error": "No active session. Please upload a resume first."
            }
        
        # Update conversation history from context if provided
        if context and "conversation_history" in context:
            self.conversation_history = context["conversation_history"]
        
        # Add user message to history
        firebase_service.add_message(self.current_session, "user", query)
        
        # Add current message to internal history
        self.conversation_history.append({
            "role": "user",
            "content": query
        })
        
        # Route query with full context
        routing = self.router.route_query(query, context)
        agent_type = routing["agent_type"]
        entities = routing["entities"]
        
        # Get current resume - full text as fallback
        latest_resume = firebase_service.get_latest_resume(self.current_session)
        full_resume_text = latest_resume["content"] if latest_resume else self.current_resume["raw_text"]
        
        # First, try to get relevant context from vector store
        relevant_context = self._get_relevant_context(query, self.current_session)
        
        # Use relevant context if available, otherwise use full resume
        resume_text = relevant_context if relevant_context else full_resume_text
        use_rag = bool(relevant_context)
        
        # Process based on agent type
        try:
            if agent_type == AgentType.COMPANY_RESEARCH:
                result = self._handle_company_research(query, resume_text, entities, full_resume_text, use_rag)
            elif agent_type == AgentType.JOB_MATCHING:
                result = self._handle_job_matching(query, resume_text, entities, full_resume_text, use_rag)
            elif agent_type == AgentType.SECTION_ENHANCEMENT:
                result = self._handle_section_enhancement(query, resume_text, entities, full_resume_text, use_rag)
            else:
                result = self._handle_general_query(query, resume_text, self.conversation_history)
            
            # If result indicates insufficient context, retry with full resume
            if result.get("needs_full_context"):
                if agent_type == AgentType.COMPANY_RESEARCH:
                    result = self._handle_company_research(query, full_resume_text, entities, full_resume_text, False)
                elif agent_type == AgentType.JOB_MATCHING:
                    result = self._handle_job_matching(query, full_resume_text, entities, full_resume_text, False)
                elif agent_type == AgentType.SECTION_ENHANCEMENT:
                    result = self._handle_section_enhancement(query, full_resume_text, entities, full_resume_text, False)
                result["used_fallback"] = True
            
            # Save response
            firebase_service.add_message(
                self.current_session,
                "assistant",
                result["response"],
                {
                    "agent_type": agent_type.value,
                    "confidence": routing["confidence"],
                    "used_rag": use_rag,
                    "used_fallback": result.get("used_fallback", False)
                }
            )
            
            # Save updated resume if changed
            if result.get("updated_resume"):
                version_id = firebase_service.save_resume_version(
                    self.current_session,
                    result["updated_resume"],
                    result.get("version_name", "Updated"),
                    {"agent_type": agent_type.value}
                )
                
                # Delete old resume chunks from vector store
                vector_store.delete_session_data(self.current_session)
                
                # Add updated resume to vector store (replaces old chunks)
                vector_store.add_resume_to_index(
                    self.current_session,
                    result["updated_resume"],
                    {
                        "version_id": version_id,
                        "version_name": result.get("version_name", "Updated")
                    }
                )
            
            # Add next steps
            result["next_steps"] = self.router.suggest_next_steps(
                agent_type,
                self.conversation_history
            )
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": result.get("response", "")
            })
            
            result["routing_info"] = {
                "agent_used": agent_type.value,
                "confidence": routing["confidence"],
                "used_rag": use_rag,
                "used_fallback": result.get("used_fallback", False)
            }
            
            # Mark if resume was updated
            if result.get("updated_resume"):
                result["resume_updated"] = True
            
            return result
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            firebase_service.add_message(self.current_session, "system", error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
    
    def _get_relevant_context(self, query: str, session_id: str, n_results: int = 3) -> Optional[str]:
        """
        Retrieve relevant context from vector store using RAG pattern
        
        Args:
            query: User query
            session_id: Session identifier
            n_results: Number of chunks to retrieve
            
        Returns:
            Combined relevant context or None
        """
        try:
            results = vector_store.search_similar_content(query, session_id, n_results)
            
            if results:
                # Combine relevant chunks
                context_pieces = [r["content"] for r in results]
                combined_context = "\n\n".join(context_pieces)
                
                # Only use if we got meaningful results (similarity threshold)
                if results[0].get("distance", 1.0) < 0.8:  # Lower distance = more similar
                    return combined_context
            
            return None
        except Exception as e:
            print(f"Error retrieving context from vector store: {e}")
            return None
    
    def _handle_company_research(self, query: str, resume_text: str, 
                                 entities: Dict, full_resume: str = None,
                                 using_rag: bool = False) -> Dict:
        """
        Handle company research requests
        
        Args:
            query: User query
            resume_text: Resume text (may be RAG context or full text)
            entities: Extracted entities
            full_resume: Full resume text for fallback
            using_rag: Whether using RAG context
        """
        company_name = entities.get("company_name")
        
        if not company_name:
            return {
                "status": "needs_info",
                "response": "I'd be happy to optimize your resume for a company! Which company are you targeting?",
                "needs_input": "company_name"
            }
        
        # Optimize for company
        try:
            raw_output = self.company_agent.optimize_for_company(
                resume_text,
                company_name
            )
            
            # Clean the output to separate resume from improvements
            cleaned = output_cleaner.extract_resume_and_improvements(raw_output)
            optimized_resume = cleaned["resume_content"]
            improvements = cleaned["improvements"]
            
            # Check if result seems incomplete (heuristic check)
            if using_rag and len(optimized_resume) < 200:
                return {
                    "needs_full_context": True
                }
            
            # Build response with improvements visible to user
            response_text = f"I've optimized your resume for {company_name}."
            if improvements:
                response_text += f"\n\n**Key Changes:**\n{improvements}"
            
            return {
                "status": "success",
                "response": response_text,
                "updated_resume": optimized_resume,
                "version_name": f"Optimized for {company_name}",
                "improvements": improvements
            }
        except Exception as e:
            # If using RAG and error occurs, request full context
            if using_rag:
                return {"needs_full_context": True}
            raise
    
    def _handle_job_matching(self, query: str, resume_text: str, 
                            entities: Dict, full_resume: str = None,
                            using_rag: bool = False) -> Dict:
        """
        Handle job matching requests
        
        Args:
            query: User query
            resume_text: Resume text (may be RAG context or full text)
            entities: Extracted entities
            full_resume: Full resume text for fallback
            using_rag: Whether using RAG context
        """
        # Check if job description is provided in query
        if len(query) > 300:  # Likely contains job description
            job_description = query
        else:
            return {
                "status": "needs_info",
                "response": "Please provide the job description you'd like to match your resume against.",
                "needs_input": "job_description"
            }
        
        try:
            # Calculate match score
            match_analysis = self.job_agent.calculate_match_score(resume_text, job_description)
            
            # Optimize resume
            raw_output = self.job_agent.match_to_job(resume_text, job_description)
            
            # Clean the output to separate resume from improvements
            cleaned = output_cleaner.extract_resume_and_improvements(raw_output)
            optimized_resume = cleaned["resume_content"]
            improvements = cleaned["improvements"]
            
            # Check if result seems incomplete
            if using_rag and len(optimized_resume) < 200:
                return {"needs_full_context": True}
            
            response = f"**Match Analysis:**\n{match_analysis['analysis']}\n\n"
            if improvements:
                response += f"**Optimization Changes:**\n{improvements}"
            else:
                response += "I've optimized your resume to better match the job description."
            
            return {
                "status": "success",
                "response": response,
                "updated_resume": optimized_resume,
                "version_name": "Job Description Match",
                "match_score": match_analysis.get("match_score", 0),
                "improvements": improvements
            }
        except Exception as e:
            if using_rag:
                return {"needs_full_context": True}
            raise
    
    def _handle_section_enhancement(self, query: str, resume_text: str, 
                                    entities: Dict, full_resume: str = None,
                                    using_rag: bool = False) -> Dict:
        """
        Handle section enhancement requests
        
        Args:
            query: User query
            resume_text: Resume text (may be RAG context or full text)
            entities: Extracted entities
            full_resume: Full resume text for fallback
            using_rag: Whether using RAG context
        """
        section_name = entities.get("section_name")
        
        if not section_name:
            return {
                "status": "needs_info",
                "response": "Which section would you like me to enhance? (e.g., Experience, Skills, Summary)",
                "needs_input": "section_name"
            }
        
        # Extract section from resume
        sections = ResumeParser.extract_sections(resume_text)
        
        if section_name not in sections:
            # Try to find similar section using simple matching first
            matched_section = None
            section_name_lower = section_name.lower()
            
            # Exact substring match
            for key in sections.keys():
                if section_name_lower in key.lower() or key.lower() in section_name_lower:
                    matched_section = key
                    break
            
            # If still not found, use LLM to intelligently match section
            if not matched_section:
                matched_section = self._match_section_with_llm(section_name, list(sections.keys()))
            
            # Final fallback: check if any section contains the word
            if not matched_section:
                for key in sections.keys():
                    key_words = key.lower().split()
                    if section_name_lower in key_words or any(word in section_name_lower for word in key_words):
                        matched_section = key
                        break
            
            if matched_section:
                section_name = matched_section
            else:
                # If still can't match, return helpful error
                return {
                    "status": "error",
                    "response": f"I couldn't find a section matching '{section_name}'. Available sections: {', '.join(sections.keys())}. Which one would you like to enhance?"
                }
        
        # Enhance section
        try:
            raw_output = self.section_agent.enhance_section(
                section_name,
                sections[section_name]
            )
            
            # Clean the output to separate section content from improvements
            cleaned = output_cleaner.extract_resume_and_improvements(raw_output)
            enhanced_section = cleaned["resume_content"]
            improvements = cleaned["improvements"]
            
            # Check if result seems incomplete
            if using_rag and len(enhanced_section) < 50:
                return {"needs_full_context": True}
            
            # Update resume with enhanced section
            sections[section_name] = enhanced_section
            from tools.resume_parser import DocumentGenerator
            updated_resume = DocumentGenerator.format_resume_sections(sections)
            
            # Build response with improvements visible to user
            response_text = f"I've enhanced your {section_name} section."
            if improvements:
                response_text += f"\n\n**Improvements Made:**\n{improvements}"
            response_text += f"\n\n**Enhanced Section:**\n{enhanced_section}"
            
            return {
                "status": "success",
                "response": response_text,
                "updated_resume": updated_resume,
                "version_name": f"Enhanced {section_name}",
                "improvements": improvements
            }
        except Exception as e:
            if using_rag:
                return {"needs_full_context": True}
            raise
    
    def _match_section_with_llm(self, requested_section: str, available_sections: List[str]) -> Optional[str]:
        """
        Use LLM to intelligently match a requested section name to available sections
        
        Args:
            requested_section: The section name user requested
            available_sections: List of actual section names in the resume
            
        Returns:
            Best matching section name or None
        """
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        import json
        
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a resume section matching expert. Match the requested section name 
                to the most appropriate available section. Consider semantic similarity and common variations.
                
                For example:
                - "work history" or "jobs" → "EXPERIENCE" or "WORK EXPERIENCE"
                - "education background" → "EDUCATION"
                - "technical skills" → "SKILLS" or "TECHNICAL SKILLS"
                - "summary" or "profile" → "SUMMARY" or "PROFILE" or "OBJECTIVE"
                
                Return ONLY the exact section name from the available sections list, or "NONE" if no good match exists."""),
                ("human", """Requested section: {requested}
Available sections: {available}

Which available section best matches the requested one? Return only the exact section name or "NONE".""")
            ])
            
            messages = prompt.format_messages(
                requested=requested_section,
                available=", ".join(available_sections)
            )
            
            response = llm.invoke(messages)
            matched = response.content.strip()
            
            # Validate the response is in available sections
            if matched in available_sections:
                return matched
            elif matched.upper() == "NONE":
                return None
            else:
                # Try case-insensitive match
                for section in available_sections:
                    if section.upper() == matched.upper():
                        return section
                return None
                
        except Exception as e:
            print(f"Error in LLM section matching: {str(e)}")
            return None
    
    def _handle_general_query(self, query: str, resume_text: str, 
                             conversation_context: List[Dict] = None) -> Dict:
        """
        Handle general queries using LLM-powered agent
        
        Args:
            query: User's question
            resume_text: Current resume content
            conversation_context: Recent conversation history
        """
        # Check if it's a greeting
        greeting_keywords = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(keyword in query.lower() for keyword in greeting_keywords):
            response = self.general_agent.handle_greeting(
                query, 
                has_resume=(resume_text is not None and len(resume_text) > 100)
            )
            return {
                "status": "success",
                "response": response
            }
        
        # Build conversation context string
        context_str = ""
        if conversation_context:
            context_str = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_context[-5:]  # Last 5 messages
            ])
        
        # Use general agent to answer query with LLM
        result = self.general_agent.answer_query(
            query=query,
            resume_text=resume_text or "No resume uploaded yet.",
            conversation_context=context_str if context_str else None
        )
        
        if result["status"] == "success":
            return {
                "status": "success",
                "response": result["answer"]
            }
        else:
            # Fallback response
            return {
                "status": "success",
                "response": """I can help you optimize your resume in several ways:

1. **Company Optimization**: Tailor your resume for specific companies (e.g., "Optimize for Google")
2. **Job Matching**: Match your resume to job descriptions and improve your score
3. **Section Enhancement**: Improve specific sections with better language and quantification

What would you like to do?"""
            }
    
    def get_session_info(self) -> Dict:
        """Get current session information"""
        if not self.current_session:
            return {"status": "no_session"}
        
        return {
            "session_id": self.current_session,
            "resume_versions": len(firebase_service.get_resume_versions(self.current_session)),
            "message_count": len(firebase_service.get_conversation_history(self.current_session)),
            "current_resume": self.current_resume
        }
    
    def get_resume_versions(self) -> List[Dict]:
        """Get all resume versions for current session"""
        if not self.current_session:
            return []
        
        return firebase_service.get_resume_versions(self.current_session)
    
    def revert_to_version(self, version_id: str) -> bool:
        """Revert to a previous resume version"""
        version = firebase_service.get_resume_by_version_id(version_id)
        if version:
            # Save as new version
            firebase_service.save_resume_version(
                self.current_session,
                version["content"],
                f"Reverted to {version['version_name']}"
            )
            return True
        return False


# Global orchestrator instance
orchestrator = ResumeOptimizationOrchestrator()
