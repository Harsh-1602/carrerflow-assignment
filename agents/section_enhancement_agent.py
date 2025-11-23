"""
Section Enhancement Agent
Improves specific resume sections with impact statements and quantification
"""
from crewai import Agent, Task, Crew
from typing import Dict, List


class SectionEnhancementAgent:
    """Agent that enhances specific resume sections"""
    
    def __init__(self, llm_config: Dict = None):
        """Initialize the Section Enhancement Agent"""
        self.llm_config = llm_config or {}
        self.agent = self._create_agent()
        self.guidelines = self._load_section_guidelines()
    
    def _load_section_guidelines(self) -> Dict[str, str]:
        """Load enhancement guidelines for different sections"""
        return {
            "SUMMARY": """
            - Start with strong action words
            - Include 2-3 key strengths
            - Quantify experience (years, achievements)
            - Tailor to target role
            - Keep to 3-4 impactful sentences
            """,
            "EXPERIENCE": """
            - Use strong action verbs (Led, Developed, Managed)
            - Quantify achievements with metrics (%, $, numbers)
            - Follow STAR method (Situation, Task, Action, Result)
            - Focus on impact and results
            - Include relevant keywords
            """,
            "SKILLS": """
            - Organize by category (Technical, Soft, Tools)
            - Prioritize most relevant skills first
            - Include proficiency levels if applicable
            - Use industry-standard terminology
            - Balance hard and soft skills
            """,
            "EDUCATION": """
            - Include degree, institution, graduation date
            - Add GPA if impressive (>3.5)
            - Highlight relevant coursework
            - Include honors and awards
            - Add certifications
            """,
            "PROJECTS": """
            - Describe problem and solution
            - Highlight technologies used
            - Quantify impact or results
            - Include links if available
            - Show leadership and collaboration
            """
        }
    
    def _get_guidelines_for_section(self, section_name: str) -> str:
        """Get guidelines for a specific section"""
        section_upper = section_name.upper()
        for key, guidelines in self.guidelines.items():
            if key in section_upper or section_upper in key:
                return f"Guidelines for {section_name}:\n{guidelines}"
        return "General guidelines: Use action verbs, quantify achievements, focus on impact."
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        return Agent(
            role="Resume Writing & Section Enhancement Specialist",
            goal="Transform resume sections into compelling, quantified statements that demonstrate impact and value",
            backstory="""You are an award-winning resume writer with expertise in crafting 
            impactful achievement statements. You excel at adding quantification, using strong 
            action verbs, and structuring content to maximize impact. You know how to transform 
            generic job descriptions into compelling accomplishment statements that catch 
            recruiters' attention.""",
            verbose=True,
            allow_delegation=False
        )
    
    def enhance_section(self, section_name: str, section_content: str, 
                       target_role: str = "", additional_context: str = "") -> str:
        """
        Enhance a specific resume section
        
        Args:
            section_name: Name of the section to enhance
            section_content: Current section content
            target_role: Optional target role for context
            additional_context: Optional additional guidance
            
        Returns:
            Enhanced section content
        """
        role_context = f"Target Role: {target_role}\n" if target_role else ""
        context = f"Additional Context: {additional_context}\n" if additional_context else ""
        guidelines = self._get_guidelines_for_section(section_name)
        
        task = Task(
            description=f"""
            Enhance the following resume section to make it more impactful and compelling.
            
            Section: {section_name}
            {role_context}{context}
            
            {guidelines}
            
            Current Content:
            {section_content}
            
            Steps:
            1. Apply the enhancement guidelines for this section type
            2. Identify weak or generic statements
            3. Add quantification and metrics where possible
            4. Replace passive language with strong action verbs
            5. Emphasize results and impact
            6. Ensure consistency in format and style
            7. Optimize for readability and ATS
            
            Provide the enhanced section with clear before/after comparisons for major changes.
            """,
            agent=self.agent,
            expected_output="""Enhanced section content with:
            1. Strong action verbs throughout
            2. Quantified achievements with specific metrics
            3. Impact-focused statements
            4. Consistent formatting
            5. Explanations of key improvements"""
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    # def add_quantification(self, experience_text: str) -> str:
    #     """
    #     Add quantification to experience statements
        
    #     Args:
    #         experience_text: Experience section text
            
    #     Returns:
    #         Quantified experience text
    #     """
    #     task = Task(
    #         description=f"""
    #         Add quantification and metrics to these experience statements.
            
    #         Current Experience:
    #         {experience_text}
            
    #         For each statement:
    #         1. Identify what can be quantified (time, money, people, %, growth)
    #         2. Add realistic metrics based on typical achievements
    #         3. Use specific numbers over vague terms
    #         4. Include context for the numbers
            
    #         Transform statements like:
    #         - "Improved system performance" → "Improved system performance by 40%, reducing load time from 3s to 1.8s"
    #         - "Managed a team" → "Led a cross-functional team of 8 developers across 3 time zones"
    #         """,
    #         agent=self.agent,
    #         expected_output="Experience statements with specific, realistic quantification added"
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
    #     return str(result)
    
    # def improve_action_verbs(self, section_content: str) -> str:
    #     """
    #     Improve action verbs in resume content
        
    #     Args:
    #         section_content: Section content to improve
            
    #     Returns:
    #         Content with stronger action verbs
    #     """
    #     task = Task(
    #         description=f"""
    #         Replace weak verbs with strong action verbs in this content.
            
    #         Current Content:
    #         {section_content}
            
    #         Replace weak verbs like:
    #         - "Responsible for" → Led, Managed, Directed, Orchestrated
    #         - "Worked on" → Developed, Built, Engineered, Architected
    #         - "Helped" → Facilitated, Enabled, Drove, Accelerated
    #         - "Did" → Executed, Implemented, Delivered, Achieved
            
    #         Ensure variety and choose verbs that best convey impact and leadership.
    #         """,
    #         agent=self.agent,
    #         expected_output="Content with strong, varied action verbs that convey impact"
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
    #     return str(result)
    
    # def create_impact_statements(self, job_duties: List[str]) -> List[str]:
    #     """
    #     Transform job duties into impact statements
        
    #     Args:
    #         job_duties: List of job duty descriptions
            
    #     Returns:
    #         List of impact-focused statements
    #     """
    #     duties_text = "\n".join([f"- {duty}" for duty in job_duties])
        
    #     task = Task(
    #         description=f"""
    #         Transform these job duties into compelling impact statements using the STAR method.
            
    #         Job Duties:
    #         {duties_text}
            
    #         For each duty:
    #         1. Identify the challenge or context (Situation/Task)
    #         2. Describe what was done (Action)
    #         3. Emphasize the outcome (Result)
    #         4. Add quantification where possible
            
    #         Example transformation:
    #         Duty: "Managed database systems"
    #         Impact: "Architected and maintained PostgreSQL databases serving 100K+ daily users, achieving 99.9% uptime and reducing query response time by 35%"
    #         """,
    #         agent=self.agent,
    #         expected_output="List of impact-focused achievement statements"
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
        
    #     # Parse result into list
    #     statements = []
    #     for line in str(result).split('\n'):
    #         line = line.strip()
    #         if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
    #             statement = line.lstrip('-•0123456789.').strip()
    #             if statement:
    #                 statements.append(statement)
        
    #     return statements if statements else [str(result)]
    
    # def optimize_skills_section(self, skills_text: str, target_role: str = "") -> str:
    #     """
    #     Optimize skills section for impact and relevance
        
    #     Args:
    #         skills_text: Current skills section
    #         target_role: Optional target role
            
    #     Returns:
    #         Optimized skills section
    #     """
    #     role_context = f" for a {target_role} role" if target_role else ""
        
    #     task = Task(
    #         description=f"""
    #         Optimize this skills section{role_context}.
            
    #         Current Skills:
    #         {skills_text}
            
    #         Steps:
    #         1. Categorize skills (Technical, Tools, Soft Skills, etc.)
    #         2. Prioritize most relevant skills first
    #         3. Remove outdated or irrelevant skills
    #         4. Add proficiency levels if helpful
    #         5. Use industry-standard terminology
    #         6. Ensure balanced coverage
            
    #         Format: Organize clearly with categories and relevant groupings.
    #         """,
    #         agent=self.agent,
    #         expected_output="Well-organized skills section with categories and prioritization"
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
    #     return str(result)
