import asyncio
import mlflow
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk

from langgraph.prebuilt import create_react_agent
from config import MLflowLoggingSettings
from tools import MCPToolsManager, MemoryManager
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Initialize memory manager
memory_manager = MemoryManager()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# Available Ollama models
OLLAMA_MODELS = [
    "qwen2.5-coder:14b",
    "granite3.3:8b", 
    "devstral:24b",
    "llama3.2:3b",
    "deepseek-r1:14b",
    "qwen3:14b",
    "gemma3:12b",
    "mistral-small3.2:24b"
]

async def main():
    # Initialize logging settings
    logging_settings = MLflowLoggingSettings(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="first steps",
        enable_system_metrics=True,
        enable_langchain_autolog=True
    )
    
    # Setup MLflow
    logging_settings.setup_mlflow()

    # Initialize tools manager
    tools_manager = MCPToolsManager()
    tools = await tools_manager.get_tools()

    # Select model from available models
    selected_model = "devstral:24b"  # Default selection
    # You can change this to any model from OLLAMA_MODELS array
    
    # Log model metadata to MLflow using the settings class
    logging_settings.log_model_and_metadata(selected_model)

    llm = ChatOllama(
        model=selected_model,
        temperature=0.1,
        num_predict=2000,
        # other params...
    ).bind_tools(tools)

    # Note: MLflow has issues logging ChatOllama models directly, so we'll skip this for now
    # mlflow.langchain.log_model(llm, "llm", registered_model_name=selected_model.replace(":", "_"))

    def chatbot(state: State):
        # Retrieve relevant long-term memories
        user_id = "default_user"  # In production, get from authentication
        last_message = state["messages"][-1] if state["messages"] else None
        query = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Get relevant memories for context
        relevant_memories = memory_manager.retrieve_relevant_memories(user_id, query)
        memory_context = memory_manager.format_memories_for_context(relevant_memories)
        
        # Enhanced messages with memory context
        enhanced_messages = state["messages"].copy()
        if memory_context:
            system_message = ChatMessage(
                role="system", 
                content=f"Context from previous interactions:\n{memory_context}\n\nUse this context to provide more personalized and informed responses."
            )
            enhanced_messages.insert(0, system_message)
        
        response = llm.invoke(enhanced_messages)
        return {"messages": [response]}

    # agent = create_react_agent(
    #     model=llm,
    #     tools=tools,  # Add the web tool here
    #     prompt="You are a helpful assistant. You can answer questions and perform tasks using the tools available to you."
    # )

    graph_builder.add_node("chatbot", chatbot)
    tool_node = await tools_manager.get_tool_node()
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_manager.route_tools,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")

    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)

    # agent_response = ""
    async def stream_graph_updates(user_input: str):
        assistant_response = ""
        async for events in graph.astream({"messages": [{"role": "user", "content": user_input}]},
                                         config={"configurable": {"thread_id": "1"}}):
            for node_name, event_data in events.items():
                print(f"[{node_name}]")
                if node_name != "chatbot":
                    continue
                # Capture assistant response for memory analysis
                if "messages" in event_data and event_data["messages"]:
                    last_message = event_data["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        assistant_response = last_message.content
                        print(f"{last_message.content}", end="", flush=True)
        
        # Save memories based on the interaction
        if assistant_response:
            await memory_manager.analyze_and_save_memories(user_input, assistant_response)
        
        return assistant_response

    traces = []
    while True:
        print("Type 'exit' to quit. Enter your message. Submit an empty line to finish.")
        user_lines = []
        exit = False
        while True:
            line = input()
            if line == 'exit' and not user_lines:
                exit = True
                break
            if line == '':
                break
            user_lines.append(line)
        if exit:
            print("Exiting chat.")
            break
        user_input = '\n'.join(user_lines)
        if not user_input.strip():
            continue

        await stream_graph_updates(user_input)

        # Print memory statistics
        user_id = "default_user"
        total_memories = memory_manager.get_memory_count(user_id)
        print(f"\n[Memory Status: {total_memories} total memories stored]")

        trace_id = mlflow.get_last_active_trace_id()
        trace = mlflow.get_trace(trace_id=trace_id)
        traces.append(trace)

    mlflow.end_run()

    # Print token usage summary using the logging settings
    logging_settings.print_token_usage_summary(traces)


if __name__ == "__main__":
    asyncio.run(main())
