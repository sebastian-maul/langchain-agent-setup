import json
import uuid
from datetime import datetime
from typing import List

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


def embed(texts: List[str]) -> List[List[float]]:
    """Simple mock embedding function - replace with actual embeddings in production"""
    import hashlib
    return [[float(int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 1000) / 1000] * 384 for text in texts]


class MemoryManager:
    """Manages long-term memory storage and retrieval for the chatbot"""
    
    def __init__(self):
        self.store = InMemoryStore(index={"embed": embed, "dims": 384})
    
    def save_semantic_memory(self, user_id: str, facts: List[str], context: str = "general") -> str:
        """Save factual information about the user (semantic memory)"""
        namespace = (user_id, "semantic", context)
        memory_id = str(uuid.uuid4())
        self.store.put(namespace, memory_id, {
            "type": "semantic",
            "facts": facts,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        return memory_id

    def save_episodic_memory(self, user_id: str, interaction: dict, task_context: str = "general") -> str:
        """Save interaction experiences (episodic memory)"""
        namespace = (user_id, "episodic", task_context)
        memory_id = str(uuid.uuid4())
        self.store.put(namespace, memory_id, {
            "type": "episodic",
            "interaction": interaction,
            "timestamp": datetime.now().isoformat(),
            "context": task_context,
            "success": interaction.get("success", True)
        })
        return memory_id

    def save_procedural_memory(self, user_id: str, instructions: str, context: str = "general"):
        """Save and update procedural instructions/preferences"""
        namespace = (user_id, "procedural")
        self.store.put(namespace, context, {
            "type": "procedural",
            "instructions": instructions,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })

    def retrieve_relevant_memories(self, user_id: str, query: str, memory_type: str = None, limit: int = 5) -> List[dict]:
        """Retrieve relevant memories based on query"""
        if memory_type:
            namespace = (user_id, memory_type)
        else:
            namespace = (user_id,)
        
        # Search for relevant memories using semantic search
        results = self.store.search(namespace, query=query, limit=limit)
        return [item.value for item in results]

    def format_memories_for_context(self, memories: List[dict]) -> str:
        """Format retrieved memories for inclusion in prompt context"""
        if not memories:
            return ""
        
        context_parts = []
        semantic_facts = []
        episodic_examples = []
        procedural_rules = []
        
        for memory in memories:
            if memory.get("type") == "semantic":
                semantic_facts.extend(memory.get("facts", []))
            elif memory.get("type") == "episodic":
                episodic_examples.append(memory.get("interaction", {}))
            elif memory.get("type") == "procedural":
                procedural_rules.append(memory.get("instructions", ""))
        
        if semantic_facts:
            context_parts.append(f"Known facts: {'; '.join(semantic_facts)}")
        if episodic_examples:
            context_parts.append(f"Past interactions: {json.dumps(episodic_examples[:3], indent=2)}")
        if procedural_rules:
            context_parts.append(f"Instructions/Preferences: {'; '.join(procedural_rules)}")
        
        return "\n".join(context_parts)

    async def analyze_and_save_memories(self, user_input: str, assistant_response: str, user_id: str = "default_user"):
        """Analyze interaction and save relevant memories"""
        
        # Create interaction record for episodic memory
        interaction = {
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "success": True  # Could be determined by feedback or other metrics
        }
        
        # Save episodic memory
        self.save_episodic_memory(user_id, interaction)
        
        # Extract facts for semantic memory (simplified - in production use LLM)
        # This is a simple heuristic - replace with LLM-based fact extraction
        if "my name is" in user_input.lower():
            name_fact = user_input.lower().split("my name is")[1].strip()
            self.save_semantic_memory(user_id, [f"User's name is {name_fact}"])
        
        if "i like" in user_input.lower() or "i prefer" in user_input.lower():
            preference = user_input.strip()
            self.save_semantic_memory(user_id, [f"User preference: {preference}"])
        
        # Could add more sophisticated fact extraction here

    def get_memory_count(self, user_id: str) -> int:
        """Get the total number of memories stored for a user"""
        return len(self.store.search((user_id,), query="", limit=1000))
