"""
Vector store service for semantic search and context retrieval
"""
from typing import List, Dict, Optional
import chromadb
from config import settings as app_settings
import uuid


class VectorStoreService:
    """
    Manages vector embeddings for semantic search
    Uses ChromaDB for local vector storage
    """
    
    def __init__(self):
        """Initialize ChromaDB client with persistent storage"""
        # Use PersistentClient to ensure data is saved to disk
        self.client = chromadb.PersistentClient(
            path=app_settings.CHROMA_PERSIST_DIRECTORY
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="resume_context",
            metadata={"description": "Resume content and conversation context"}
        )
    
    def add_resume_to_index(self, session_id: str, resume_text: str, 
                           metadata: Optional[Dict] = None) -> str:
        """
        Add resume content to vector store
        
        Args:
            session_id: Session identifier
            resume_text: Resume text content
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        doc_id = f"{session_id}_{uuid.uuid4()}"
        
        # Split resume into chunks for better retrieval
        chunks = self._split_into_chunks(resume_text)
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "session_id": session_id,
                "chunk_index": i,
                "document_id": doc_id,
                **(metadata or {})
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return doc_id
    
    def search_similar_content(self, query: str, session_id: Optional[str] = None,
                              n_results: int = 5) -> List[Dict]:
        """
        Search for similar content in vector store
        
        Args:
            query: Search query
            session_id: Optional session filter
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        where_filter = {"session_id": session_id} if session_id else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else None
                })
        
        return formatted_results
    
    def get_session_context(self, session_id: str) -> List[str]:
        """
        Get all context for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of context chunks
        """
        results = self.collection.get(
            where={"session_id": session_id}
        )
        
        return results['documents'] if results else []
    
    def delete_session_data(self, session_id: str) -> bool:
        """
        Delete all data for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        try:
            # Get all IDs for this session
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
            
            return True
        except Exception as e:
            print(f"Error deleting session data: {e}")
            return False
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in words
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def update_document(self, doc_id: str, new_text: str, 
                       metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing document
        
        Args:
            doc_id: Document identifier
            new_text: New text content
            metadata: Optional updated metadata
            
        Returns:
            Success status
        """
        try:
            # Delete old chunks
            results = self.collection.get(
                where={"document_id": doc_id}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
            
            # Add new chunks
            chunks = self._split_into_chunks(new_text)
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "chunk_index": i,
                    "document_id": doc_id,
                    **(metadata or {})
                })
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False


# Global instance
vector_store = VectorStoreService()
