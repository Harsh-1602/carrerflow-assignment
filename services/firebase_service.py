"""
Firebase service for storing conversation history and resume versions
"""
import json
from typing import Dict, List, Optional
from datetime import datetime
import uuid


class FirebaseService:
    """
    Manages conversation history and resume versions
    Note: This is a simplified in-memory implementation for demo purposes.
    In production, this would connect to actual Firebase.
    """
    
    def __init__(self):
        # In-memory storage (replace with actual Firebase in production)
        self.conversations = {}
        self.resume_versions = {}
        self.users = {}
    
    def create_session(self, user_id: str = None) -> str:
        """
        Create a new conversation session
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        self.conversations[session_id] = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {}
        }
        
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Optional[Dict] = None) -> bool:
        """
        Add a message to conversation history
        
        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        if session_id not in self.conversations:
            return False
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[session_id]["messages"].append(message)
        return True
    
    def get_conversation_history(self, session_id: str, 
                                 limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve conversation history
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]["messages"]
        
        if limit:
            return messages[-limit:]
        return messages
    
    def save_resume_version(self, session_id: str, resume_content: str, 
                           version_name: str = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Save a resume version
        
        Args:
            session_id: Session identifier
            resume_content: Resume text content
            version_name: Optional version name
            metadata: Optional metadata
            
        Returns:
            Version ID
        """
        version_id = str(uuid.uuid4())
        
        if session_id not in self.resume_versions:
            self.resume_versions[session_id] = []
        
        version = {
            "version_id": version_id,
            "session_id": session_id,
            "content": resume_content,
            "version_name": version_name or f"Version {len(self.resume_versions[session_id]) + 1}",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.resume_versions[session_id].append(version)
        return version_id
    
    def get_resume_versions(self, session_id: str) -> List[Dict]:
        """
        Get all resume versions for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of resume versions
        """
        return self.resume_versions.get(session_id, [])
    
    def get_latest_resume(self, session_id: str) -> Optional[Dict]:
        """
        Get the latest resume version
        
        Args:
            session_id: Session identifier
            
        Returns:
            Latest resume version or None
        """
        versions = self.get_resume_versions(session_id)
        return versions[-1] if versions else None
    
    def get_resume_by_version_id(self, version_id: str) -> Optional[Dict]:
        """
        Get a specific resume version by ID
        
        Args:
            version_id: Version identifier
            
        Returns:
            Resume version or None
        """
        for session_versions in self.resume_versions.values():
            for version in session_versions:
                if version["version_id"] == version_id:
                    return version
        return None
    
    def update_session_metadata(self, session_id: str, metadata: Dict) -> bool:
        """
        Update session metadata
        
        Args:
            session_id: Session identifier
            metadata: Metadata to update
            
        Returns:
            Success status
        """
        if session_id not in self.conversations:
            return False
        
        self.conversations[session_id]["metadata"].update(metadata)
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
        
        if session_id in self.resume_versions:
            del self.resume_versions[session_id]
        
        return True
    
    def export_conversation(self, session_id: str) -> Optional[str]:
        """
        Export conversation as JSON
        
        Args:
            session_id: Session identifier
            
        Returns:
            JSON string or None
        """
        if session_id not in self.conversations:
            return None
        
        export_data = {
            "conversation": self.conversations[session_id],
            "resume_versions": self.resume_versions.get(session_id, [])
        }
        
        return json.dumps(export_data, indent=2)


# Global instance
firebase_service = FirebaseService()
