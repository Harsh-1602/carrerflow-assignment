"""
Streamlit Chat Interface for Resume Optimization System
"""
import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from agents.orchestrator import orchestrator
from config import settings
import time

# Page configuration
st.set_page_config(
    page_title="Resume Optimizer - AI-Powered Resume Enhancement",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .system-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        font-style: italic;
    }
    .version-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #fafafa;
    }
    .metric-card {
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = orchestrator
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resume_uploaded' not in st.session_state:
    st.session_state.resume_uploaded = False
if 'resume_updated' not in st.session_state:
    st.session_state.resume_updated = False
if 'latest_resume' not in st.session_state:
    st.session_state.latest_resume = ""

def display_message(role: str, content: str, metadata: dict = None):
    """Display a chat message"""
    if role == "user":
        css_class = "user-message"
        icon = "👤"
    elif role == "assistant":
        css_class = "assistant-message"
        icon = "🤖"
    else:
        css_class = "system-message"
        icon = "ℹ️"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.capitalize()}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)
    
    if metadata and st.session_state.get('show_debug', False):
        with st.expander("🔍 Debug Info"):
            st.json(metadata)

def handle_file_upload(uploaded_file):
    """Handle resume file upload"""
    try:
        # Save uploaded file
        upload_dir = Path(settings.UPLOAD_DIR)
        file_path = upload_dir / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Start optimization session
        with st.spinner("Analyzing your resume..."):
            result = st.session_state.orchestrator.start_session(str(file_path))
        
        if result["status"] == "success":
            st.session_state.session_active = True
            st.session_state.resume_uploaded = True
            st.session_state.session_info = result
            
            # Add welcome message
            welcome_msg = f"""✅ Resume loaded successfully!
            
**File:** {result['resume_info']['file_name']}
**Words:** {result['resume_info']['word_count']}
**Sections Found:** {', '.join(result['resume_info']['sections'])}

I'm ready to help you optimize your resume! You can:
- Optimize for a specific company (e.g., "Optimize for Google")
- Match against a job description
- Enhance specific sections (e.g., "Improve my experience section")

What would you like to do?"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg,
                "timestamp": datetime.now()
            })
            
            st.success("Resume loaded successfully!")
            st.rerun()
        else:
            st.error(f"Error loading resume: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

def process_user_input(user_input: str):
    """Process user message"""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Build conversation context from recent messages
    context = {
        "conversation_history": [],
        "message_count": len(st.session_state.messages)
    }
    
    # Include last 5 messages for context (excluding the one we just added)
    recent_messages = st.session_state.messages[-6:-1] if len(st.session_state.messages) > 1 else []
    for msg in recent_messages:
        context["conversation_history"].append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"].isoformat() if isinstance(msg["timestamp"], datetime) else str(msg["timestamp"])
        })
    
    # Process query with context
    with st.spinner("🤔 Thinking..."):
        response = st.session_state.orchestrator.process_query(user_input, context)
    
    # Add assistant response
    if response["status"] == "success":
        content = response.get("response", "I've processed your request.")
        
        # Check if resume was updated
        if response.get("resume_updated"):
            st.session_state.resume_updated = True
            st.session_state.latest_resume = response.get("updated_resume", "")
        
        # Add next steps
        if response.get("next_steps"):
            content += "\n\n**Suggested next steps:**\n"
            for step in response["next_steps"][:2]:
                content += f"- {step}\n"
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(),
            "metadata": response.get("routing_info", {})
        })
    elif response["status"] == "needs_info":
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("response", "I need more information."),
            "timestamp": datetime.now()
        })
    else:
        st.session_state.messages.append({
            "role": "system",
            "content": f"Error: {response.get('error', 'Unknown error')}",
            "timestamp": datetime.now()
        })

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📄 AI Resume Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Resume Enhancement with Multi-Agent AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Session Info")
        
        if not st.session_state.session_active:
            st.info("👆 Upload your resume to get started")
            
            # File uploader
            st.header("📤 Upload Resume")
            uploaded_file = st.file_uploader(
                "Choose your resume (PDF or DOCX)",
                type=["pdf", "docx"],
                help="Upload your current resume to start optimizing"
            )
            
            if uploaded_file is not None:
                if st.button("🚀 Start Optimization", type="primary"):
                    handle_file_upload(uploaded_file)
        else:
            # Session info
            session_info = st.session_state.session_info
            st.success("✅ Session Active")
            
            st.metric("File", session_info['resume_info']['file_name'])
            st.metric("Word Count", session_info['resume_info']['word_count'])
            
            # Resume versions
            st.header("📚 Resume Versions")
            versions = st.session_state.orchestrator.get_resume_versions()
            
            if versions:
                st.info(f"Total versions: {len(versions)}")
                
                for i, version in enumerate(reversed(versions[-5:])):  # Show last 5
                    with st.expander(f"Version {len(versions) - i}: {version['version_name']}"):
                        st.text(f"Created: {version['created_at'][:19]}")
                        
                        # Revert button
                        if st.button(f"🔄 Revert to this version", key=f"revert_{version['version_id']}"):
                            if st.session_state.orchestrator.revert_to_version(version['version_id']):
                                # Update session state to reflect revert
                                st.session_state.latest_resume = version['content']
                                st.session_state.resume_updated = True
                                st.success("Reverted successfully!")
                                st.rerun()
                        
                        # Download buttons for each version
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # PDF download for this version
                            if st.button("📄 PDF", key=f"pdf_{version['version_id']}", use_container_width=True):
                                from tools.resume_parser import DocumentGenerator
                                import tempfile
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                    pdf_path = DocumentGenerator.create_pdf(
                                        version['content'],
                                        tmp.name,
                                        version['version_name']
                                    )
                                    
                                    with open(pdf_path, 'rb') as f:
                                        pdf_data = f.read()
                                
                                st.download_button(
                                    label="⬇️ Download PDF",
                                    data=pdf_data,
                                    file_name=f"resume_v{len(versions) - i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_pdf_{version['version_id']}"
                                )
                        
                        with col2:
                            # DOCX download for this version
                            if st.button("📝 DOCX", key=f"docx_{version['version_id']}", use_container_width=True):
                                from tools.resume_parser import DocumentGenerator
                                import tempfile
                                
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                                    docx_path = DocumentGenerator.create_docx(
                                        version['content'],
                                        tmp.name
                                    )
                                    
                                    with open(docx_path, 'rb') as f:
                                        docx_data = f.read()
                                
                                st.download_button(
                                    label="⬇️ Download DOCX",
                                    data=docx_data,
                                    file_name=f"resume_v{len(versions) - i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"dl_docx_{version['version_id']}"
                                )
            else:
                st.info("No versions yet")
            
            # Download Resume Section
            st.header("📥 Download Resume")
            if st.session_state.get('resume_updated', False) and st.session_state.get('latest_resume'):
                st.success("✅ Updated resume available!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate PDF
                    if st.button("📄 Download PDF", use_container_width=True):
                        from tools.resume_parser import DocumentGenerator
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            pdf_path = DocumentGenerator.create_pdf(
                                st.session_state.latest_resume,
                                tmp.name,
                                "Optimized Resume"
                            )
                            
                            with open(pdf_path, 'rb') as f:
                                pdf_data = f.read()
                            
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=pdf_data,
                                file_name=f"resume_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="download_pdf"
                            )
                
                with col2:
                    # Generate DOCX
                    if st.button("📝 Download DOCX", use_container_width=True):
                        from tools.resume_parser import DocumentGenerator
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                            docx_path = DocumentGenerator.create_docx(
                                st.session_state.latest_resume,
                                tmp.name
                            )
                            
                            with open(docx_path, 'rb') as f:
                                docx_data = f.read()
                            
                            st.download_button(
                                label="⬇️ Download DOCX",
                                data=docx_data,
                                file_name=f"resume_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="download_docx"
                            )
            else:
                st.info("Process your resume to enable downloads")
            
            # Options
            st.header("⚙️ Options")
            st.session_state.show_debug = st.checkbox("Show debug info", value=False)
            
            if st.button("🔄 New Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # About
        st.markdown("---")
        st.markdown("""
        ### About
        This AI-powered system uses specialized agents to:
        - Research companies and optimize for culture fit
        - Match resumes to job descriptions
        - Enhance sections with impact statements
        
        Built with CrewAI & Streamlit
        """)
    
    # Main chat interface
    if st.session_state.session_active:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(
                    message["role"],
                    message["content"],
                    message.get("metadata")
                )
        
        # Chat input
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Ask me to optimize your resume...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.button("Send", type="primary", use_container_width=True)
            
            # Quick actions
            st.markdown("**Quick Actions:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏢 Optimize for Company"):
                    st.session_state.quick_action = "optimize_company"
            with col2:
                if st.button("📋 Match Job Description"):
                    st.session_state.quick_action = "match_job"
            with col3:
                if st.button("✨ Enhance Section"):
                    st.session_state.quick_action = "enhance_section"
            
            # Handle quick actions
            if 'quick_action' in st.session_state:
                action = st.session_state.quick_action
                del st.session_state.quick_action
                
                if action == "optimize_company":
                    company = st.text_input("Enter company name:")
                    if company:
                        process_user_input(f"Optimize my resume for {company}")
                        st.rerun()
                elif action == "match_job":
                    jd = st.text_area("Paste job description:")
                    if jd:
                        process_user_input(f"Match my resume to this job description: {jd}")
                        st.rerun()
                elif action == "enhance_section":
                    section = st.selectbox("Choose section:", 
                                          ["Experience", "Skills", "Summary", "Education", "Projects"])
                    if st.button("Enhance"):
                        process_user_input(f"Enhance my {section} section")
                        st.rerun()
            
            # Process input
            if send_button and user_input:
                process_user_input(user_input)
                st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to AI Resume Optimizer! 👋
        
        This intelligent system helps you create the perfect resume by:
        
        ### 🎯 What We Can Do
        
        **1. Company Optimization** 🏢
        - Research target company culture and values
        - Adapt your resume language and tone
        - Emphasize relevant achievements
        
        **2. Job Description Matching** 📋
        - Analyze job requirements
        - Calculate your match score
        - Restructure to highlight relevant experience
        - Optimize for ATS systems
        
        **3. Section Enhancement** ✨
        - Add quantification and metrics
        - Use powerful action verbs
        - Create impact statements
        - Improve overall quality
        
        ### 🚀 Get Started
        
        Upload your resume in the sidebar to begin!
        """)
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - "Optimize my resume for Google"
            - "Match my resume to this job description: [paste JD]"
            - "Improve my experience section with better metrics"
            - "Enhance my skills section for a data scientist role"
            - "Make my summary more impactful"
            """)

if __name__ == "__main__":
    main()
