import json
import uuid
from datetime import datetime
from typing import List

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


def embed(texts: List[str]) -> List[List[float]]:
    """Improved mock embedding function - creates different embeddings for different content"""
    import hashlib
    import re
    
    embeddings = []
    for text in texts:
        # Clean and normalize text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Create a simple but more meaningful embedding
        # Using word hashes at different positions to create diversity
        embedding = [0.0] * 384
        
        for i, word in enumerate(words[:20]):  # Use first 20 words
            word_hash = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            # Spread the word influence across multiple dimensions
            base_idx = (word_hash % 19) * 20  # 19 groups of 20 dimensions
            for j in range(20):
                if base_idx + j < 384:
                    embedding[base_idx + j] += (word_hash % 1000) / 1000.0
        
        # Add text length and character variety features
        embedding[380] = min(len(text) / 100.0, 1.0)  # Length feature
        embedding[381] = len(set(text.lower())) / 26.0  # Character diversity
        embedding[382] = len(words) / 50.0 if words else 0.0  # Word count feature
        embedding[383] = text.count('?') + text.count('!') * 0.5  # Punctuation feature
        
        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x/norm for x in embedding]
        
        embeddings.append(embedding)
    
    return embeddings


class MemoryManager:
    """Manages long-term memory storage and retrieval for the chatbot"""
    
    def __init__(self):
        self.store = InMemoryStore(index={"embed": embed, "dims": 384})
    
    def save_semantic_memory(self, user_id: str, facts: List[str], context: str = "general") -> str:
        """Save factual information about the user (semantic memory)"""
        namespace = (user_id, "semantic", context)
        memory_id = str(uuid.uuid4())
        
        # Create searchable content
        searchable_content = " ".join(facts)
        
        self.store.put(namespace, memory_id, {
            "type": "semantic",
            "facts": facts,
            "searchable_content": searchable_content,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        return memory_id

    def save_episodic_memory(self, user_id: str, interaction: dict, task_context: str = "general") -> str:
        """Save interaction experiences (episodic memory)"""
        namespace = (user_id, "episodic", task_context)
        memory_id = str(uuid.uuid4())
        
        # Create searchable content from the interaction
        user_input = interaction.get("user_input", "")
        assistant_response = interaction.get("assistant_response", "")
        searchable_content = f"{user_input} {assistant_response}"
        
        self.store.put(namespace, memory_id, {
            "type": "episodic",
            "interaction": interaction,
            "searchable_content": searchable_content,
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
            "searchable_content": instructions,
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
        results = self.store.search(namespace, query=query, limit=limit * 2)  # Get more results to filter
        
        # Filter results by a simple relevance threshold if we have more than the limit
        if len(results) > limit:
            # Calculate simple similarity scores based on keyword overlap
            filtered_results = []
            query_words = set(query.lower().split())
            
            for item in results:
                memory = item.value
                # Get searchable content
                searchable_content = memory.get('searchable_content', '')
                if not searchable_content:
                    # Fallback to other content
                    if memory.get('type') == 'semantic':
                        searchable_content = ' '.join(memory.get('facts', []))
                    elif memory.get('type') == 'episodic':
                        interaction = memory.get('interaction', {})
                        searchable_content = f"{interaction.get('user_input', '')} {interaction.get('assistant_response', '')}"
                    elif memory.get('type') == 'procedural':
                        searchable_content = memory.get('instructions', '')
                
                content_words = set(searchable_content.lower().split())
                overlap = len(query_words.intersection(content_words))
                relevance_score = overlap / max(len(query_words), 1)
                
                if relevance_score > 0.1:  # At least 10% word overlap
                    filtered_results.append((item, relevance_score))
            
            # Sort by relevance score and take top results
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            results = [item for item, score in filtered_results[:limit]]
        
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
        memory_id = self.save_episodic_memory(user_id, interaction)
        
        # Extract facts for semantic memory (simplified - in production use LLM)
        # This is a simple heuristic - replace with LLM-based fact extraction
        if "my name is" in user_input.lower():
            name_fact = user_input.lower().split("my name is")[1].strip()
            semantic_id = self.save_semantic_memory(user_id, [f"User's name is {name_fact}"])
        
        if "i like" in user_input.lower() or "i prefer" in user_input.lower():
            preference = user_input.strip()
            semantic_id = self.save_semantic_memory(user_id, [f"User preference: {preference}"])
        
        # Could add more sophisticated fact extraction here

    def get_memory_count(self, user_id: str) -> int:
        """Get the total number of memories stored for a user"""
        return len(self.store.search((user_id,), query="", limit=1000))
