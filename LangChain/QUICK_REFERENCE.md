# ⚡ LANGCHAIN QUICK REFERENCE - Interview Cheat Sheet



A Runnable is a standard interface (contract) in LangChain.
Any component that follows this interface must be able to take input, process it, and return output.



## 6-Second Concept Definitions

### Core Components
| Concept | Definition | Example |
|---------|-----------|---------|
| **LangChain** | Framework for LLM app composition | `chain = prompt \| model \| parser` |
| **Runnable** | Any input → output component | Prompts, Models, Tools, Functions |
| **Chain** | Composable Runnables linked by pipes | `A \| B \| C` |
| **Model** | LLM or Chat model API wrapper | `ChatOpenAI(model="gpt-4")` |
| **Prompt** | Template with variable injection | `PromptTemplate(template="...", variables=[...])` |
| **Parser** | Converts output to structured data | `StrOutputParser()`, `PydanticOutputParser()` |
| **Agent** | LLM that reasons and chooses tools | Uses tools dynamically based on task |
| **Tool** | Callable function for agent use | `@tool` decorated functions |
| **Memory** | Persists conversation context | `ConversationBufferMemory()` |
| **RAG** | Augments LLM with external knowledge | Load → Split → Embed → Retrieve → Generate |

---

## Essential Patterns (Copy-Paste Ready)

### Pattern 1: Simple Chain
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(template="Tell me about {topic}", input_variables=["topic"])
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "Python"})
```

### Pattern 2: Structured Output
```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Result(BaseModel):
    sentiment: str
    confidence: float

llm = ChatOpenAI()
structured_llm = llm.with_structured_output(Result)
result = structured_llm.invoke("I love this!")
# Returns: Result(sentiment="positive", confidence=0.95)
```

### Pattern 3: Chat with History
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])

history = [
    HumanMessage("Hi"), 
    AIMessage("Hello!")
]
messages = template.format_messages(history=history, query="How are you?")
```

### Pattern 4: RAG Pipeline
```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load & Split & Embed
loader = TextLoader("doc.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# Retrieve
results = vector_store.similarity_search("query", k=5)
```

### Pattern 5: Basic Agent
```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search tool"""
    return f"Results for {query}"

llm = ChatOpenAI()
tools = [search]
prompt = ChatPromptTemplate.from_messages([...])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "Search for Python"})
```

---

## Common Interview Answers

### Q: What is LangChain?
**Answer:** A Python framework that simplifies building LLM applications by providing abstractions for common tasks: prompting, model interaction, tool usage, memory management, and output parsing. It works with multiple LLM providers (OpenAI, Google, HuggingFace) with a unified interface.

### Q: What's a Runnable?
**Answer:** LangChain's standard interface for any component that takes input and produces output. All Runnables support `.invoke()`, `.stream()`, `.batch()` methods and can be chained using `|` operator. Examples: prompts, models, parsers, tools, custom functions.

### Q: Chains vs Agents?
**Answer:**
- **Chains:** Predetermined flow (A→B→C), rigid, predictable, fast
- **Agents:** Dynamic flow, LLM decides path, adaptive, slower

### Q: How does RAG work?
**Answer:** 
1. Load documents (TextLoader, PDFLoader, etc.)
2. Split into chunks (RecursiveCharacterTextSplitter)
3. Create embeddings (OpenAIEmbeddings)
4. Store in vector DB (Chroma, Pinecone)
5. Retrieve relevant docs via similarity search
6. Inject as context into LLM prompt
7. Generate grounded response

### Q: What's Tool Binding vs Tool Calling?
**Answer:**
- **Binding:** Register tools as JSON schemas with LLM (`llm.bind_tools([tool1, tool2])`)
- **Calling:** LLM decides which tool to use and with what params
- **Execution:** Tool actually runs, result injected back to LLM

### Q: How do Output Parsers work?
**Answer:** Convert raw LLM text into structured data. Types:
- StrOutputParser: Clean text
- JsonOutputParser: Parse JSON
- PydanticOutputParser: Type-validated models
- with_structured_output(): Built into model (preferred)

### Q: What's Memory in LangChain?
**Answer:** Persists conversation context since LLM API calls are stateless.
- BufferMemory: Store all messages
- WindowMemory: Keep last K messages
- SummarizedMemory: Condense old messages

### Q: Memory vs Context Window?
**Answer:**
- **Context Window:** Model's internal capacity (e.g., GPT-4 = 8K-128K tokens)
- **Memory:** LangChain component that manages what to send to the model
Memory helps you stay within context window.

### Q: What are common RAG failure modes?
**Answer:**
1. Poor chunking → Fragmented context
2. Bad retrieval → Wrong docs returned
3. LLM ignores context → Uses training knowledge (hallucination)
4. Token overflow → Context exceeds window
5. Stale documents → Outdated knowledge

### Q: How do you prevent hallucinations?
**Answer:**
1. Use RAG (ground in real data)
2. Use structured output (validate format)
3. Add constraints to prompts ("Only answer from provided context")
4. Use reranking (validate retrieved docs)
5. Implement guardrails (check factual claims)

---

## Architecture Flowcharts

### Simple Chain Flow
```
User Input
    ↓
PromptTemplate (format)
    ↓
LLM/ChatModel (generate)
    ↓
OutputParser (structure)
    ↓
Application Output
```

### RAG Flow
```
User Query
    ↓
Embedding (query → vector)
    ↓
VectorDB Search (k-NN)
    ↓
Retrieved Documents
    ↓
Context Injection → LLM Prompt
    ↓
LLM Generation (grounded)
    ↓
Output
```

### Agent Flow
```
User Goal
    ↓
LLM (reason, plan)
    ↓
Choose Tool
    ↓
Execute Tool
    ↓
Get Result
    ↓
LLM Sees Result
    ↓
Decide: Continue? or Done?
    ↓
Final Response
```

---

## Decision Trees

### Which Parser to Use?
```
Need structured data?
├─ NO → Use StrOutputParser
└─ YES
   ├─ JSON format OK? 
   │  ├─ YES → Use JsonOutputParser
   │  └─ NO → Use PydanticOutputParser
   └─ Modern LLM (GPT-4, Claude 3+)?
      ├─ YES → Use with_structured_output()
      └─ NO → Use PydanticOutputParser
```

### Which Memory to Use?
```
Conversation length?
├─ Short (<10 messages) → BufferMemory
├─ Medium (10-100) → WindowMemory
├─ Long (100+) → SummarizedMemory
└─ Persistent (across sessions) → Database + Custom Memory
```

### Which Vector DB?
```
Scale?
├─ Small (<10K docs) → Chroma (in-memory)
├─ Medium (10K-1M) → Pinecone / Weaviate
└─ Large (1M+) → Milvus / Elasticsearch
   ├─ Managed? YES → Pinecone
   └─ Self-hosted? YES → Weaviate / Milvus
```

### Chain vs Agent?
```
Fixed process? 
├─ YES → Chain (faster, predictable)
└─ NO → Agent (flexible, adaptive)
   ├─ Simple logic? YES → RunnableBranch
   └─ Complex reasoning? YES → Agent
```

---

## Code Snippets by Problem

### Problem: Extract structured data from text
```python
from pydantic import BaseModel
class DataModel(BaseModel):
    field1: str
    field2: int

llm = ChatOpenAI()
parser = llm.with_structured_output(DataModel)
result = parser.invoke("text to parse")
```

### Problem: Search external knowledge
```python
from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(chunks, embeddings)
relevant = vector_store.similarity_search("query", k=5)
context = "\n".join([doc.page_content for doc in relevant])
```

### Problem: Handle multiple steps
```python
chain1 = prompt1 | llm
chain2 = prompt2 | llm
chain3 = prompt3 | llm
full_chain = chain1 | chain2 | chain3
result = full_chain.invoke(input)
```

### Problem: Parallel execution
```python
parallel = {
    "path_a": chain_a,
    "path_b": chain_b,
    "path_c": chain_c
}
result = parallel.invoke(input)
# Returns {"path_a": ..., "path_b": ..., "path_c": ...}
```

### Problem: Conditional logic
```python
branch = RunnableBranch(
    (lambda x: condition1(x), chain_a),
    (lambda x: condition2(x), chain_b),
    default_chain
)
result = branch.invoke(input)
```

### Problem: Multiple turns conversation
```python
messages = [HumanMessage("First"), AIMessage("Reply")]
for turn in range(5):
    response = llm.invoke(messages)
    messages.append(response)
```



---

## Interview Do's & Don'ts

### ✅ DO
- ✅ Mention LangChain uses abstractions (key feature)
- ✅ Explain why RAG matters (hallucination reduction)
- ✅ Show understanding of token economics (cost, latency)
- ✅ Mention production considerations (error handling, monitoring)
- ✅ Discuss tradeoffs (speed vs quality, cost vs accuracy)
- ✅ Code examples should compile and make sense

### ❌ DON'T
- ❌ Confuse LLM with Chat Model (different concepts)
- ❌ Say "LLMs have memory" (they don't, need Memory component)
- ❌ Forget about token limits (token limits matter!)
- ❌ Assume tools execute automatically (explicit execution needed)
- ❌ Ignore error handling (production-grade code includes it)
- ❌ Overcomplicate explanation (use analogies, simple language)

---

## One-Liner Definitions (Memorize These!)

```
Runnable = Any component that takes input → output and supports .invoke()
Chain = Runnable composition using pipe operator (|)
Model = LLM/ChatModel API wrapper
Prompt = Template with variable injection
Parser = Converts output to structured format
Agent = LLM that reasons about which tools to use
Tool = Callable function accessible to agents
Memory = Component that persists conversation context
RAG = Augmenting LLM generation with external document retrieval
Embedding = Vector representation of text for semantic search
```

---

## Common Mistakes to Avoid

| Mistake | Why Wrong | Fix |
|---------|----------|-----|
| Using fstrings instead of PromptTemplate | No ecosystem integration | Use PromptTemplate |
| Assuming LLM remembers | LLMs are stateless | Use Memory component |
| Not validating output | Garbage in, garbage out | Use Pydantic parsers |
| Ignoring token limits | Expensive/fails silently | Monitor token usage |
| Not handling tool errors | Production breaks | Implement error handling |
| RAG without reranking | Poor retrieval quality | Add ContextualCompression |
| Infinite agent loops | Wastes tokens, never finishes | Set max_iterations |
| No fallbacks | Single point of failure | Use .with_fallbacks() |

---

## Performance / Optimization Tips

| Goal | Strategy |
|------|----------|
| **Lower Latency** | Use `.stream()` instead of `.invoke()` |
| **Lower Cost** | Use cheaper models (gpt-3.5 vs gpt-4) |
| **Better Quality** | Add RAG context, use reranking |
| **Handle Long Docs** | Use semantic chunking + windowed memory |
| **Scale to Many Docs** | Use cloud vector DB (Pinecone) + batch operations |
| **Prevent Hallucination** | Add RAG, structure output, validate |

---

## Quick Debugging Checklist

- [ ] Does the prompt template format correctly?
- [ ] Does the model respond to the prompt?
- [ ] Does the parser handle the model's output format?
- [ ] Are tools registered correctly (binding)?
- [ ] Do tools have clear docstrings and type hints?
- [ ] Is memory being updated between turns?
- [ ] Is RAG retrieving relevant documents?
- [ ] Are there token limit violations?
- [ ] Is error handling in place?
- [ ] Are API keys correctly configured?

---

## 5-Minute Pre-Interview Prep

**5 min before interview:**
- [ ] Mentally rehearse: LLM vs Chat Model difference
- [ ] Remember: LLMs are stateless (need Memory)
- [ ] Recall: Runnable is standard interface
- [ ] Think: Chain = A | B | C composition
- [ ] Quick: RAG = Load → Split → Embed → Retrieve → Generate
- [ ] Mental model: Agent = Reason → Choose Tool → Execute → Reflect

**If asked to code:**
- Start with prompt template
- Add model (ChatOpenAI)
- Add parser
- Compose with pipes
- Test with .invoke()

---

**Last updated:** 2026-04-10  
**Coverage:** All major LangChain concepts  
**Purpose:** Interview quick reference  
**Format:** Memorizable, code-ready patterns
