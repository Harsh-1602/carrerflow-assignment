"""
Company Research and Optimization Agent
Researches target companies and optimizes resumes for company culture
"""
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import Dict
import os


class CompanyResearchAgent:
    """Agent that researches companies and optimizes resumes"""
    
    def __init__(self, llm_config: Dict = None):
        """Initialize the Company Research Agent"""
        self.llm_config = llm_config or {}
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        # Initialize web research tools
        search_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        
        return Agent(
            role="Company Research & Resume Optimization Specialist",
            goal="Research target companies and optimize resumes to match company culture, values, and hiring patterns",
            backstory="""You are an expert career consultant with deep knowledge of corporate 
            cultures across industries. You excel at researching companies and understanding 
            what they value in candidates. You can analyze company hiring patterns and adapt 
            resume content to resonate with specific company cultures while maintaining authenticity.""",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool, scrape_tool]
        )
    
    def optimize_for_company(self, resume_text: str, company_name: str, 
                            additional_context: str = "") -> str:
        """
        Optimize resume for a specific company
        
        Args:
            resume_text: Current resume content
            company_name: Target company name
            additional_context: Optional additional context
            
        Returns:
            Optimized resume text
        """
        task = Task(
            description=f"""
            Research {company_name} and optimize the following resume to match their culture and values.
            
            Company: {company_name}
            Additional Context: {additional_context}
            
            Current Resume:
            {resume_text}
            
            Steps:
            1. Research {company_name}'s culture, values, and hiring patterns
            2. Identify key qualities and skills they prioritize
            3. Restructure and reword the resume to emphasize relevant experience
            4. Adjust language tone to match company culture (e.g., formal vs casual)
            5. Highlight achievements that align with company values
            6. Ensure the optimized resume maintains authenticity
            
            Provide the complete optimized resume with clear improvements marked.
            """,
            agent=self.agent,
            expected_output="""A fully optimized resume tailored to the target company, with:
            1. Restructured sections emphasizing relevant experience
            2. Language and tone matching company culture
            3. Highlighted achievements aligned with company values
            4. Clear explanations of key changes made"""
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    # def analyze_company_fit(self, resume_text: str, company_name: str) -> Dict:
    #     """
    #     Analyze how well the resume fits the company
        
    #     Args:
    #         resume_text: Current resume content
    #         company_name: Target company name
            
    #     Returns:
    #         Analysis with fit score and recommendations
    #     """
    #     task = Task(
    #         description=f"""
    #         Analyze how well this resume fits {company_name}'s requirements and culture.
            
    #         Company: {company_name}
            
    #         Resume:
    #         {resume_text}
            
    #         Provide:
    #         1. Overall fit score (0-100)
    #         2. Strengths that align with the company
    #         3. Areas for improvement
    #         4. Specific recommendations for optimization
    #         """,
    #         agent=self.agent,
    #         expected_output="""Analysis report with:
    #         - Fit score and justification
    #         - 3-5 key strengths
    #         - 3-5 areas for improvement
    #         - Actionable recommendations"""
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
        
    #     return {
    #         "company": company_name,
    #         "analysis": str(result),
    #         "agent_type": "company_research"
    #     }
