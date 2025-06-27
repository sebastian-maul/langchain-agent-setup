import asyncio
import mlflow
import mlflow.langchain

from langchain_ollama import ChatOllama
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.prebuilt import create_react_agent

async def main():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.langchain.autolog()
    mlflow.set_experiment("first steps")

    mlflow.start_run()

    client = MultiServerMCPClient(
        {
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "transport": "stdio",
            },
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/home/anni/dev/python/redo",
                ],
                "transport": "stdio",
            },
            "time": {
                "command": "uvx",
                "args": ["mcp-server-time", "--local-timezone=Europe/Berlin"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()

    llm = ChatOllama(
        # model="qwen2.5-coder:14b",
        # model="granite3.3:8b",
        # model="mistral-small3.1:latest",
        model="devstral:24b", # +
        # model="qwen3:14b",
        # model="deepseek-r1:14b", # no tools
        temperature=0.1,
        num_predict=2000,
        # other params...
    ).bind_tools(tools)

    agent = create_react_agent(
        model=llm,
        tools=tools,  # Add the web tool here
        prompt="You are a helpful assistant. You can answer questions and perform tasks using the tools available to you."
    )

    messages = [
        ChatMessage(role="control", content="thinking"),
    ]

    traces = []
    print("Type 'exit' to quit. Enter your message. Submit an empty line to finish.")
    while True:
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
        messages.append(HumanMessage(user_input))
        print("Agent:", end=" ")
        agent_response = ""
        async for token, metadata in agent.astream({"messages": messages}, stream_mode="messages"):
            print(f"[{type(token)}] {token.content}")
            agent_response += token.content
        print("\n")
        messages.append(ChatMessage(role="assistant", content=agent_response))
        trace_id = mlflow.get_last_active_trace_id()
        trace = mlflow.get_trace(trace_id=trace_id)
        traces.append(trace)

    mlflow.end_run()

    total_in = 0
    total_out = 0
    total_usage = 0

    # Print the token usage
    for trace in traces:
        total_in += trace.info.token_usage['input_tokens']
        total_out += trace.info.token_usage['output_tokens']
        total_usage += trace.info.token_usage['total_tokens']

    print("== Total token usage: ==")
    print(f"  Input tokens: {total_in}")
    print(f"  Output tokens: {total_out}")
    print(f"  Total tokens: {total_usage}")

if __name__ == "__main__":
    asyncio.run(main())
