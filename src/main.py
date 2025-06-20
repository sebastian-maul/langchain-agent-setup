from autogen_core.models import UserMessage, ModelInfo
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
import yaml
import asyncio
import os


def main():
    # Load Ollama model config
    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][0]

    # Provide ModelInfo for custom model
    granite_model_info = ModelInfo(
        family="unknown",  # Not a known family, so use 'unknown'
        function_calling=True,
        json_output=True,
        structured_output=True,
        vision=False
    )

    # Create Ollama model client with model_info
    ollama_client = OllamaChatCompletionClient(
        model=model_cfg["model_name"],
        base_url=model_cfg["model_server"],
        temperature=model_cfg.get("temperature", 0.2),
        max_tokens=model_cfg.get("max_tokens", 2048),
        request_timeout=model_cfg.get("request_timeout", 120),
        model_info=granite_model_info
    )

    # Create ChromaDB vector memory for RAG
    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="autogen_docs",
            persistence_path=os.path.expanduser("~/.chromadb_autogen"),
            k=3,
            score_threshold=0.4,
        )
    )

    # Create RAG assistant agent
    rag_assistant = AssistantAgent(
        name="rag_assistant",
        model_client=ollama_client,
        memory=[rag_memory],
    )

    async def chat_loop():
        print("Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            stream = rag_assistant.run_stream(task=user_input)
            await Console(stream)
        await ollama_client.close()
        await rag_memory.close()

    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
