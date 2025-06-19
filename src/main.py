from autogen_core.models import UserMessage, ModelInfo
from autogen_ext.models.ollama import OllamaChatCompletionClient
import yaml
import asyncio


def main():
    # Load Ollama model config
    with open("model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][0]

    # Provide ModelInfo for custom model
    granite_model_info = ModelInfo(
        family="unknown",  # Not a known family, so use 'unknown'
        function_calling=False,
        json_output=False,
        structured_output=False,
        vision=False
    )

    # Create Ollama model client with model_info
    ollama_client = OllamaChatCompletionClient(
        model=model_cfg["model_name"],
        base_url=model_cfg["model_server"],
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 2048),
        request_timeout=model_cfg.get("request_timeout", 120),
        model_info=granite_model_info
    )

    async def chat_loop():
        print("Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            result = await ollama_client.create([
                UserMessage(content=user_input, source="user")
            ])
            print(f"Ollama: {result.content.strip()}")
        await ollama_client.close()

    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
