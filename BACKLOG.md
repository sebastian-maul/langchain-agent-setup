# MVP Backlog for AI Agent Team (LangChain, LangGraph, MLflow)

- [ ] Set up project structure and environment (Python, uv, dependencies)
- [ ] Integrate MLflow for LLM model and prompt versioning
- [ ] Register and load LLM models from MLflow for agent use
- [ ] Store and manage agent role prompts in MLflow
- [ ] Implement agent memory (short-term: in-memory, long-term: pgvector/Postgres)
- [ ] Evaluate and integrate Chroma for document storage (compare with pgvector)
- [ ] Implement RAG pipeline for project-related documents (web scraping, PDF, markdown ingestion)
- [ ] Integrate web search tool for agents
- [ ] Integrate database tool (Postgres/pgvector) for agents
- [ ] Integrate file and folder manipulation tools for agents
- [ ] Integrate git operation tools for agents
- [ ] Define minimal set of agent roles (e.g., Researcher, Coder, Reviewer)
- [ ] Implement hybrid agent workflow (sequential + collaborative)
- [ ] Add logging for agent actions and decisions (MLflow tracking)
- [ ] Document MVP setup and usage in README

Links:
- https://python.langchain.com/docs/introduction/
- https://langchain-ai.github.io/langgraph/
- https://mlflow.org/docs/latest/genai/

use the fetch tool to fetch the content of this page https://langchain-ai.github.io/langgraph/agents/context/ and give me a summary of it's content.


I have a prompt that I presented to you.
when fetchin websites you should stop fetching when less than the requested size arrived.
rewrite the prompt to add the new requirement and give me a rating on how effective it would be, when you were presented with it.
here is the prompt:
"""You are a helpful assistant. You can answer questions and perform tasks using the tools available to you.
When fetching webpages request chunks of size 5000. remember to set the start_index and increase it on subsequent requests."""


fetch the content of /home/anni/dev/python/redo/README.md, fetch the content of each page, and give me a one or two paragraph summary for each page.

1. fetch the content from https://langchain-ai.github.io/langgraph/agents/memory/
2. read the content of file /home/anni/dev/python/redo/src/main.py
3. only show the lines with changes and show the diff of your changes to the original.
4. what changes to the main.py do you propose to have short-term memory functionality for the agents?