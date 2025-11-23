"""
Job Description Matching Agent
Analyzes job descriptions and optimizes resumes for specific roles
"""
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, SerperDevTool
from typing import Dict, List


class JobMatchingAgent:
    """Agent that matches resumes to job descriptions"""
    
    def __init__(self, llm_config: Dict = None):
        """Initialize the Job Matching Agent"""
        self.llm_config = llm_config or {}
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent"""
        # Initialize tools for job analysis
        file_tool = FileReadTool()
        search_tool = SerperDevTool()
        
        return Agent(
            role="Job Description Matching & ATS Optimization Expert",
            goal="Analyze job descriptions and restructure resumes to maximize match scores while maintaining authenticity",
            backstory="""You are a seasoned technical recruiter and ATS (Applicant Tracking System) 
            expert with years of experience matching candidates to roles. You understand how to 
            parse job descriptions, identify key requirements, and restructure resumes to highlight 
            relevant experience. You know exactly how to optimize for ATS systems while keeping 
            content genuine and impactful.""",
            verbose=True,
            allow_delegation=False,
            tools=[file_tool, search_tool]
        )
    
    def match_to_job(self, resume_text: str, job_description: str) -> str:
        """
        Optimize resume to match job description
        
        Args:
            resume_text: Current resume content
            job_description: Target job description
            
        Returns:
            Optimized resume text
        """
        task = Task(
            description=f"""
            Analyze this job description and restructure the resume to maximize match score.
            
            Job Description:
            {job_description}
            
            Current Resume:
            {resume_text}
            
            Steps:
            1. Extract all key requirements from the job description
            2. Identify matching experience and skills in the resume
            3. Restructure resume sections to highlight relevant experience first
            4. Add or emphasize keywords from the job description naturally
            5. Quantify achievements where possible
            6. Ensure ATS-friendly formatting and keyword placement
            7. Fill any skill gaps with relevant transferable skills
            
            Provide the optimized resume with explanations of key changes.
            """,
            agent=self.agent,
            expected_output="""An ATS-optimized resume that:
            1. Highlights relevant experience for the specific role
            2. Includes key terms from job description naturally
            3. Quantifies achievements with metrics
            4. Emphasizes matching skills prominently
            5. Explains major restructuring decisions"""
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
    
    def calculate_match_score(self, resume_text: str, job_description: str) -> Dict:
        """
        Calculate how well resume matches job description
        
        Args:
            resume_text: Current resume content
            job_description: Target job description
            
        Returns:
            Match analysis with score and gaps
        """
        task = Task(
            description=f"""
            Calculate detailed match score between this resume and job description.
            
            Job Description:
            {job_description}
            
            Resume:
            {resume_text}
            
            Provide:
            1. Overall match score (0-100) with breakdown by category:
               - Skills match (%)
               - Experience match (%)
               - Education match (%)
               - Keyword coverage (%)
            2. Matched qualifications (what the candidate has)
            3. Skill gaps (what's missing)
            4. Recommendations to improve score
            """,
            agent=self.agent,
            expected_output="""Match analysis report with:
            - Overall score and category breakdowns
            - List of matched qualifications
            - Identified skill gaps
            - Prioritized recommendations"""
        )
        
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "match_score": self._extract_score(str(result)),
            "analysis": str(result),
            "agent_type": "job_matching"
        }
    
    # def suggest_missing_skills(self, resume_text: str, job_description: str) -> List[str]:
    #     """
    #     Identify missing skills from job description
        
    #     Args:
    #         resume_text: Current resume content
    #         job_description: Target job description
            
    #     Returns:
    #         List of missing skills
    #     """
    #     task = Task(
    #         description=f"""
    #         Identify skills mentioned in the job description that are missing from the resume.
            
    #         Job Description:
    #         {job_description}
            
    #         Resume:
    #         {resume_text}
            
    #         List all missing skills that would strengthen the application.
    #         """,
    #         agent=self.agent,
    #         expected_output="A prioritized list of missing skills with importance ratings"
    #     )
        
    #     crew = Crew(
    #         agents=[self.agent],
    #         tasks=[task],
    #         verbose=True
    #     )
        
    #     result = crew.kickoff()
        
    #     # Parse the result to extract skills
    #     skills = []
    #     for line in str(result).split('\n'):
    #         line = line.strip()
    #         if line and (line.startswith('-') or line.startswith('•')):
    #             skill = line.lstrip('-•').strip()
    #             if skill:
    #                 skills.append(skill)
        
    #     return skills
    
    def _extract_score(self, analysis_text: str) -> int:
        """Extract numeric score from analysis text"""
        import re
        
        # Look for patterns like "score: 85" or "85%" or "Score: 85/100"
        patterns = [
            r'score[:\s]+(\d+)',
            r'(\d+)%',
            r'(\d+)\s*/\s*100'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        return 0  # Default if no score found
