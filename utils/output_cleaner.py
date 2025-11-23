"""
Output Cleaner Utility
Separates agent explanations from actual resume content
"""
from openai import OpenAI
from config import settings
from typing import Dict


class OutputCleaner:
    """Cleans agent outputs to extract resume content and explanations"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def extract_resume_and_improvements(self, agent_output: str) -> Dict[str, str]:
        """
        Extract resume content and improvement explanations from agent output
        
        Args:
            agent_output: Raw output from CrewAI agent
            
        Returns:
            Dict with 'resume_content' and 'improvements' keys
        """
        prompt = f"""You are a text extraction specialist. An AI agent has optimized a resume and provided output that contains BOTH:
1. The actual resume content (should be formatted professionally)
2. Explanations of changes, improvements, or analysis

Your task: Separate these two parts.

Rules:
- Extract ONLY the actual resume content (name, contact, experience, education, skills, etc.)
- Do NOT include any meta-commentary like "Here's the optimized resume", "Key changes:", "Improvements made:", etc.
- The resume should be clean, professional text ready to use
- Put ALL explanations, analysis, and improvement notes in the 'improvements' section
- If the output is ONLY resume content with no explanations, return empty string for improvements
- If the output is ONLY analysis with no resume content, return empty string for resume_content

Agent Output:
{agent_output}

Respond in this EXACT format:
===RESUME_CONTENT===
[extracted resume content here, or empty if none]
===IMPROVEMENTS===
[extracted improvements/explanations here, or empty if none]
===END==="""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise text extraction assistant. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the structured output
            resume_content = ""
            improvements = ""
            
            if "===RESUME_CONTENT===" in result and "===IMPROVEMENTS===" in result:
                parts = result.split("===RESUME_CONTENT===")[1].split("===IMPROVEMENTS===")
                resume_content = parts[0].strip()
                improvements = parts[1].split("===END===")[0].strip()
            else:
                # Fallback: assume entire output is resume content
                resume_content = result.strip()
            
            return {
                "resume_content": resume_content,
                "improvements": improvements
            }
            
        except Exception as e:
            print(f"Error in output cleaning: {e}")
            # Fallback: return original output as resume content
            return {
                "resume_content": agent_output,
                "improvements": ""
            }
    
    def clean_resume_only(self, agent_output: str) -> str:
        """
        Extract only the resume content, discarding improvements
        
        Args:
            agent_output: Raw output from CrewAI agent
            
        Returns:
            Clean resume content
        """
        result = self.extract_resume_and_improvements(agent_output)
        return result["resume_content"]


# Global instance
output_cleaner = OutputCleaner()
