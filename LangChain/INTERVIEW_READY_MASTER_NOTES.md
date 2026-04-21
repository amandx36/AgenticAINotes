# 🚀 LangChain Complete Study Notes - Interview Ready

---

# TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Core LangChain Components](#core-components)
3. [Chains & Runnables](#chains--runnables)
4. [Models (LLMs vs Chat Models)](#models-llms-vs-chat-models)
5. [Tools & Agents](#tools--agents)
6. [RAG System](#rag-system)
7. [Structured Output & Parsers](#structured-output--parsers)
8. [Memory Management](#memory-management)
9. [Prompts & Prompt Engineering](#prompts--prompt-engineering)
10. [Interview Prep Questions](#interview-prep-questions)

---

## 📌 PROJECT OVERVIEW

### What is This Project?
Your **LangChainNotes** project is a comprehensive learning repository for **LangChain** — an open-source framework that simplifies building applications powered by Large Language Models (LLMs). It demonstrates how to integrate LLMs with external tools, knowledge bases, and data processing pipelines.

### Why It Exists
LangChain abstracts the complexity of working with different LLM providers (Google, OpenAI, HuggingFace, etc.). Instead of writing custom API integrations for each provider, you use LangChain's unified interface. This project documents best practices for:
- Chaining operations (Sequential, Parallel, Conditional)
- Building Retrieval-Augmented Generation (RAG) systems
- Creating intelligent Agents
- Handling structured outputs

### Where It Fits in the Architecture
```
User Query
    ↓
Prompt Template (Dynamic Input)
    ↓
LLM/Chat Model (API Call)
    ↓
Output Parser (Format Conversion)
    ↓
Chain Execution (Sequential/Parallel/Conditional)
    ↓
Tool Usage / Vector DB Search (External Knowledge)
    ↓
Agent Reasoning & Execution (Decision Making)
    ↓
Final Response
```

---

## 🧠 CORE COMPONENTS BREAKDOWN

### 1. **Models** 🤖
**What:** The interface to interact with AI models.
**Types:** 
- **LLMs**: Text-in → Text-out (Stateless)
- **Embedding Models**: Text-in → Vector-out (For semantic search)
- **Chat Models**: Message sequence-in → Chat message-out (Stateful conversations)

**Why Important?** 
LangChain provides a unified interface so you write code once and swap providers (GPT-4 → Claude → LLaMA) without changing logic.

**Interview Definition:**
> "Models are the core AI engines. LangChain abstracts multiple LLM providers (OpenAI, Google, Hugging Face) under one interface so you don't rewrite code for each provider."

---

### 2. **Prompts** 📝
**What:** Instructions/queries given to the LLM to guide its output.
**Types:**
- **Static Prompts**: Fixed text with no variables
- **Dynamic Prompts**: Use templates with placeholders for user input
- **ChatPromptTemplate**: Manages multi-message conversations

**Key Advantages Over f-strings:**
- Default validation
- Proper integration with LangChain ecosystem
- Message placeholder support for conversation history

**Interview Definition:**
> "Prompts are templates that guide LLM behavior. Dynamic prompts use placeholders instead of f-strings for flexibility and LangChain ecosystem integration."

**Example:**
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Generate 5 facts about {topic}",
    input_variables=["topic"]
)
result = prompt.invoke({"topic": "India"})
```

---

### 3. **Chains** ⛓️
**What:** Combine multiple components (prompts, models, parsers) in sequence or parallel.
**Types:**
- **Sequential Chain**: A → B → C (output of A becomes input to B)
- **Parallel Chain**: Run A and B simultaneously, merge results
- **Conditional Chain**: If condition → Path A, else → Path B

**Syntax:**
```python
chain = prompt | model | parser  # Pipe operator for chaining
result = chain.invoke({"topic": "India"})
```

**Interview Definition:**
> "Chains compose Runnables (prompts, models, parsers) together. Sequential chains process data step-by-step, while parallel chains execute concurrently."

---

### 4. **Runnables** 🎯
**What:** LangChain's standard interface for composable units of work.
**Key Feature:** Every Runnable supports `.invoke()`, `.stream()`, `.batch()` and async versions.

**Types of Runnables:**
1. **RunnableLambda**: Wraps Python functions
   ```python
   RunnableLambda(lambda x: x + 10)
   ```

2. **RunnableSequence**: Step-by-step pipeline
   ```python
   chain = runnable1 | runnable2 | runnable3
   ```

3. **RunnableParallel**: Execute simultaneously
   ```python
   {"a": runnable1, "b": runnable2}
   ```

4. **RunnablePassthrough**: Pass input forward unchanged
   ```python
   RunnablePassthrough()
   ```

5. **RunnableBranch**: Conditional logic
   ```python
   if condition → Branch A, else → Branch B
   ```

**Interview Definition:**
> "Runnables are the Lego bricks of LangChain. Any component (prompt, model, tool) that processes input → output is a Runnable and can be composed via the pipe operator."

---

### 5. **Memory** 🧠💾
**What:** Stores conversation history since LLM API calls are stateless.

**Types:**
- **ConversationalBufferMemory**: Stores all messages (simple but memory-intensive)
- **ConversationBufferWindowMemory**: Keeps only last N messages (sliding window)
- **SummarizedBufferWindowMemory**: Summarizes old messages to save memory
- **Custom Memory**: Build your own logic

**Interview Definition:**
> "Memory persists conversation context. Buffer memory stores messages; window memory keeps only recent messages; summarized memory summarizes old messages to save tokens."

---

### 6. **Output Parsers** 🎨
**What:** Convert LLM responses into structured, usable formats.

**Types:**
| Parser | Input | Output | Use Case |
|--------|-------|--------|----------|
| **StringOutputParser** | Text | Clean string | Simple text extraction |
| **JSONOutputParser** | Text | JSON object | API responses |
| **PydanticOutputParser** | Text | Pydantic model | Type-safe with validation |
| **StructuredOutputParser** | Text | Structured format | Limited validation |

**Example:**
```python
from pydantic import BaseModel, Field

class Review(BaseModel):
    sentiment: str = Field(description="pos/neg/neutral")
    summary: str = Field(description="Brief summary")
    pros: list[str] = Field(description="List of positives")
    cons: list[str] = Field(description="List of negatives")

model = ChatOpenAI()
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("Write a product review...")
```

**Interview Definition:**
> "Output parsers convert raw LLM text into structured data. Pydantic parsers ensure type safety and validation; JSON parsers handle flexible structures; string parsers do minimal processing."

---

### 7. **Tools & Agents** 🔧🤖

#### **Tools**
**What:** Python functions wrapped so LLMs can understand and call them.

**How It Works:**
1. Wrap function with `@tool` decorator
2. Add docstring & type hints
3. LLM receives JSON schema of the tool
4. When needed, LLM calls the tool with parameters

**Types of Tools:**
- Built-in: DuckDuckGo search, Shell commands, APIs
- Custom: Wrap your business logic

**Tool Calling Process:**
```
Tool Binding → LLM knows what tools exist & their schemas
    ↓
Tool Calling → LLM decides which tool to use & parameters (not execute yet)
    ↓
Tool Execution → Actual function runs with suggested parameters
    ↓
Tool Message → LLM receives result
```

**Example:**
```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.invoke("Obama's first name?")  # Returns search results

# Tool Binding example:
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([search])  # Register tool with LLM
```

**Interview Definition:**
> "Tools are functions wrapped for LLM consumption. Tool binding registers them with the LLM; tool calling is when LLM decides to use them; tool execution actually runs the function."

#### **Agents**
**What:** Intelligent systems that reason about goals, break them into steps, and use tools to achieve them.

**Features:**
- ✅ Goal-driven (understands high-level objectives)
- ✅ Autonomous planning (breaks work into steps)
- ✅ Tool using (leverages external APIs, databases)
- ✅ Context aware (maintains reasoning history)
- ✅ Adaptive (adjusts to new information)

**Design Patterns:**
- **ReAct**: Reasoning + Acting pattern
- **Self-ask with Search**: Ask yourself questions then search
- **Tool use with reflection**: Execute tool, reflect, iterate

**Interview Definition:**
> "Agents are LLM-powered systems that autonomously plan tasks, choose tools, execute them, and adapt. Unlike simple chains, agents think and decide dynamically."

---

## ⛓️ CHAINS & RUNNABLES DEEP DIVE

### Sequential Chain
```python
# Problem: Generate facts → Translate to Spanish → Count words
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "India"})
print(result)

# Visualize:
chain.get_graph().print_ascii()
```

**Flow:**
```
Input: {"topic": "India"}
  ↓ (PromptTemplate)
Formatted Prompt: "Generate 5 interesting facts about India"
  ↓ (GoogleGenerativeAI)
Raw Response: "Fact 1: India has... Fact 2: India also..."
  ↓ (StrOutputParser)
Output: Clean string with facts
```

---

### Parallel Chain
```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    facts=prompt | model,
    translation=translator_prompt | translator_model,
    word_count=RunnableLambda(lambda x: {"word_count": len(x.split())})
)

result = parallel_chain.invoke({"topic": "India"})
# Returns: {"facts": "...", "translation": "...", "word_count": {...}}
```

**Flow:**
```
        ↙ Prompt A → Model A
Input →               → Combine Results
        ↘ Prompt B → Model B
```

---

### Conditional Chain
```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["topic"] == "AI", ai_chain),
    (lambda x: x["topic"] == "Biology", bio_chain),
    default_chain  # Fallback
)

result = branch.invoke({"topic": "AI"})  # Uses ai_chain
```

---

## 🤖 MODELS: LLMs vs Chat Models

### LLMs (Large Language Models - Base Models)
**Input:** Text
**Output:** Text
**Memory:** None (stateless)
**Use:** Free-form generation, summarization, translation
**Examples:** GPT-3, Llama 2-7B, Mistral-7B

```python
from langchain.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
response = llm("Write a poem about coding")
```

### Chat Models (Instruction-Tuned)
**Input:** Message sequence with roles (system, user, assistant)
**Output:** Chat message
**Memory:** Structured conversation support
**Use:** Multi-turn conversations, chatbots, virtual assistants
**Examples:** GPT-4, Claude 3, LLaMA-2-Chat

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-4")
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is LangChain?")
]
response = chat.invoke(messages)
```

### Key Differences

| Feature | LLMs | Chat Models |
|---------|------|------------|
| **Training** | General text corpora | Chat datasets + instruction tuning |
| **Input Format** | Raw text | Structured messages |
| **Role Awareness** | ❌ No | ✅ Yes (system/user/assistant) |
| **Conversation Context** | Manual management | Built-in support |
| **Use Case** | Text generation | Conversational AI |
| **Modern Usage** | ≈15% | ≈85% (preferred) |

**Interview Definition:**
> "LLMs are base models for free-form generation. Chat models are instruction-tuned variants optimized for conversations with structured message support. Chat models are the modern standard."

---

## 🔧 TOOLS & AGENTS ARCHITECTURE

### Tool Definition Flow
```
1. Define Function
   ↓
2. Add @tool Decorator + Docstring + Type Hints
   ↓
3. LLM Receives JSON Schema (not actual function)
   ↓
4. LLM Decides When/How to Call
   ↓
5. Tool Executes with Suggested Parameters
   ↓
6. Result Returns to LLM
```

### Built-in Tool Examples
```python
# DuckDuckGo Search
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# Shell Command Execution
from langchain_community.tools import ShellTool
shell = ShellTool()

# API Wrapper
from langchain.tools import tool

# Database Queries
@tool
def query_database(query: str) -> str:
    """Query the product database"""
    # Your DB logic
    return results
```

### Custom Tool Example
```python
from langchain_core.tools import tool

@tool
def calculate_total_price(item_count: int, price_per_item: float) -> float:
    """Calculate total price for items.
    
    Args:
        item_count: Number of items
        price_per_item: Price per unit
    
    Returns:
        Total price
    """
    return item_count * price_per_item

# Get JSON Schema (what LLM sees)
print(calculate_total_price.args_schema.model_json_schema())

# Output:
# {
#   "name": "calculate_total_price",
#   "description": "Calculate total price for items.",
#   "input_schema": {
#     "type": "object",
#     "properties": {
#       "item_count": {"type": "integer"},
#       "price_per_item": {"type": "number"}
#     }
#   }
# }
```

### Agent Execution Loop
```
1. User provides goal: "Find weather in NYC and book a hotel"
   ↓
2. Agent (LLM) plans: "Need weather API → then hotel booking API"
   ↓
3. Agent calls Tool 1: weather_api(location="NYC")
   ↓
4. Get Result: {"temp": 72, "condition": "sunny"}
   ↓
5. Agent calls Tool 2: book_hotel(location="NYC", weather="sunny")
   ↓
6. Get Result: {"booking_id": "12345", "price": "$120"}
   ↓
7. Agent synthesizes: "Weather is 72°F and sunny. I've booked Hotel XYZ for $120"
```

**Interview Definition:**
> "Agents think → plan → choose tools → execute → reflect. Unlike chains that follow fixed paths, agents dynamically decide which tools to use based on the problem."

---

## 📚 RAG SYSTEM (Retrieval-Augmented Generation)

### What is RAG?
Combines **LLM knowledge** (learned patterns) with **external knowledge** (documents, databases) for accurate, current answers.

**Why?** LLMs have knowledge cutoffs and hallucinate. RAG grounds answers in real data.

### RAG Pipeline

```
User Query: "What is the refund policy?"
    ↓
Step 1: DOCUMENT LOADING
  - TextLoader, PDFLoader, WebLoader, CSVLoader
  - Standardized format: Document {page_content, metadata}
    ↓
Step 2: TEXT SPLITTING
  - Length-based: Fixed character/token windows
  - Structure-based: Recursive by paragraph → line → word
  - Semantic-based: Split by meaning preservation
    ↓
Step 3: EMBEDDING
  - Convert text chunks to vectors using embedding model
  - "The refund policy is..." → [0.234, -0.891, 0.456, ...]
    ↓
Step 4: VECTOR STORAGE
  - Store embeddings with metadata in vector database
  - Chroma, Pinecone, Weaviate, Milvus
    ↓
Step 5: RETRIEVAL
  - Convert query to embedding
  - Find similar vectors (k-nearest neighbors)
  - Retrieve top-K relevant documents
    ↓
Step 6: CONTEXT INJECTION
  - Add retrieved documents to prompt as context
    ↓
Step 7: LLM GENERATION
  - LLM uses context to generate grounded answer:
  - "Based on the policy, the refund period is 30 days..."
```

### Implementation Example

#### 1. Load Documents
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")
documents = loader.load()

# Structure:
# Document {
#     page_content: "Cricket is a bat-and-ball sport...",
#     metadata: {"source": "cricket.txt", "line": 1}
# }

print(documents[0].page_content)
print(documents[0].metadata)
```

#### 2. Split Text
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,           # Max 500 characters per chunk
    chunk_overlap=50,         # 50 char overlap between chunks
    separators=["\n\n", "\n", " "]  # Split by paragraph → line → word
)

chunks = splitter.split_documents(documents)
# Result: [Document(page_content="...", metadata={}), ...]
```

#### 3. Create Embeddings & Store
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

# Store in Chroma (in-memory vector DB)
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="cricket_knowledge"
)
```

#### 4. Retrieve Relevant Docs
```python
query = "Famous cricket players"

# Get top-3 most similar documents
relevant_docs = vector_store.similarity_search(query, k=3)

for doc in relevant_docs:
    print(doc.page_content)
    print(f"Similarity: {doc.metadata}")
```

#### 5. Generate Answer with Context
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4")

template = """Use the following context to answer the user's question:

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Combine context + question
context_str = "\n".join([doc.page_content for doc in relevant_docs])

result = chat.invoke({
    "context": context_str,
    "question": query
})

print(result.content)
```

### Text Splitting Strategies

#### Length-Based
```python
from langchain.text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(long_text)
# Simple but may split mid-sentence
```

#### Recursive (Smart)
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Order matters!
)
# Result:
# - First tries to split by "\n\n" (paragraphs)
# - If still too big, splits by "\n" (lines)
# - If still too big, splits by " " (words)
# - If still too big, splits by "" (characters)
```

#### Document Structure-Based
```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# Respects markdown structure
```

#### Semantic-Based (Advanced)
```python
from langchain.text_splitters import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    separators=[".", " "],
    embeddings=OpenAIEmbeddings()
)
# Splits while preserving semantic meaning
```

### Document Loaders

| Loader | Input | Output | Use Case |
|--------|-------|--------|----------|
| **TextLoader** | .txt files | Document objects | Plain text knowledge bases |
| **PDFLoader** | .pdf files | Document per page | Research papers, manuals (text only) |
| **WebBaseLoader** | URLs | Web content | Newsletter archives, docs |
| **CSVLoader** | .csv files | Structured data | Tabular data, databases |
| **DirectoryLoader** | Folder path | Multiple docs | Batch loading |

### Interview Definition
> "RAG augments LLMs with external knowledge. Documents are loaded, chunked, embedded, stored in vectors DB, retrieved via similarity, and injected into prompts for grounded generation."

---

## 📊 STRUCTURED OUTPUT & PARSERS

### Why Structured Output?
LLMs return raw text. Applications need **typed, validated data**.

**Use Cases:**
- Data extraction (invoice → structured data)
- API building (LLM → JSON → API)
- Agent coordination (tool inputs need specific types)

### Implementation Methods

#### Method 1: with_structured_output() (Recommended)
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class ProductReview(BaseModel):
    sentiment: str = Field(description="positive/negative/neutral")
    rating: int = Field(description="1-5 rating")
    summary: str = Field(description="Concise summary")
    pros: list[str] = Field(description="List of positives")
    cons: list[str] = Field(description="List of negatives")

model = ChatOpenAI(model="gpt-4")
structured_model = model.with_structured_output(ProductReview)

result = structured_model.invoke("""
The Samsung Galaxy S24 Ultra is an absolute powerhouse! The 
processor makes everything lightning fast. The camera is stunning 
with incredible zoom. However, the weight makes one-handed use 
difficult, and the price of $1,300 is steep.
""")

# Result: ProductReview(
#   sentiment="positive",
#   rating=4,
#   summary="Powerful flagship phone with stunning camera but high price",
#   pros=["Fast processor", "200MP camera", "Battery life"],
#   cons=["Heavy", "Expensive", "Bloatware"]
# )

print(result.sentiment)  # Direct attribute access
```

#### Method 2: PydanticOutputParser
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Review(BaseModel):
    sentiment: str
    summary: str

parser = PydanticOutputParser(pydantic_object=Review)

# Include format instructions in prompt
prompt_template = """Extract review data:
{format_instructions}
Review: {review_text}"""

prompt = ChatPromptTemplate.from_template(prompt_template)

chain = prompt | model | parser

result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "review_text": "Great product, love it!"
})
```

#### Method 3: JSON Output Parser
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class ReviewSchema(BaseModel):
    sentiment: str
    score: int

parser = JsonOutputParser(pydantic_object=ReviewSchema)

result = parser.parse('{"sentiment": "positive", "score": 5}')
# Result: ReviewSchema(sentiment="positive", score=5)
```

### Output Parser Comparison

| Parser | Input | Output | Validation | Use Case |
|--------|-------|--------|-----------|----------|
| **StrOutputParser** | Raw text | String | ❌ None | Simple text extraction |
| **JSONOutputParser** | JSON text | Python dict | ⚠️ Partial | Flexible structures |
| **PydanticOutputParser** | Text | Pydantic model | ✅ Full | Type-safe extraction |
| **StructuredOutputParser** | Text | Structured format | ⚠️ Limited | Custom formats |

### Interview Definition
> "Structured output converts raw LLM text into typed, validated data. Pydantic parsers enforce strict schemas; JSON parsers allow flexibility; string parsers do minimal processing."

---

## 💬 PROMPTS & PROMPT ENGINEERING

### Prompt Types

#### 1. Static Prompts
```python
prompt = "Translate to French: Hello, how are you?"
```

#### 2. Dynamic Prompts with Templates
```python
from langchain_core.prompts import PromptTemplate

template = "Complete the sentence: {sentence_start}"
prompt = PromptTemplate(
    template=template,
    input_variables=["sentence_start"]
)

result = prompt.format(sentence_start="The sky is")
# Output: "Complete the sentence: The sky is"
```

#### 3. Chat Prompts with Messages
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in coding."),
    ("human", "{user_query}"),
])

result = chat_template.format_messages(user_query="What is Python?")
```

#### 4. Chat Prompts with History
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="chat_history"),  # Dynamic history
    ("human", "{user_input}"),
])

# Usage:
history = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!")
]

result = chat_template.format_messages(
    chat_history=history,
    user_input="How are you?"
)
```

### Message Types in LangChain

```python
from langchain_core.messages import (
    SystemMessage,    # System instructions
    HumanMessage,     # User input
    AIMessage,        # Assistant response
    ToolMessage       # Tool execution result
)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What's 2+2?"),
    AIMessage(content="The answer is 4"),
]

response = chat_model.invoke(messages)
```

### MessagePlaceholders for Dynamic Conversation

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Without MessagesPlaceholder - requires manual history management:
template = """System: You are helpful
History: {history}
User: {query}"""

# With MessagesPlaceholder - automatic history injection:
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    MessagesPlaceholder(variable_name="history"),  # Injected at runtime
    ("human", "{query}"),
])

# At runtime:
formatted = chat_template.format_messages(
    history=[HumanMessage("Hi"), AIMessage("Hello")],
    query="Tell me a joke"
)
```

### Few-Shot Prompting
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

result = few_shot_prompt.format(input="big")
# Output shows examples then asks for antonym of "big"
```

### Interview Definition
> "Prompts are templates that guide LLM behavior. Static prompts are fixed; dynamic prompts use variables; chat prompts manage conversations with MessagePlaceholders for history injection."

---

## 🧠 MEMORY MANAGEMENT

### Memory Types & Use Cases

#### 1. ConversationalBufferMemory
Stores **all messages** in conversation history.

**Pros:**
- Complete conversation context
- No information loss

**Cons:**
- High token usage for long conversations
- Scales poorly

**When to use:** Short conversations, demos

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hi, what's your name?"},
    {"output": "I'm Claude"}
)
print(memory.buffer)
# Output: "Human: Hi, what's your name?\nAI: I'm Claude"
```

#### 2. ConversationBufferWindowMemory
Keeps only the **last N messages** (sliding window).

**Pros:**
- Bounded memory usage
- Recency bias (recent context matters more)

**Cons:**
- Loses early conversation context

**When to use:** Long conversations

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 messages
# Older messages are automatically forgotten
```

#### 3. ConversationSummarizedBufferMemory
**Summarizes** old messages to save tokens.

**Pros:**
- Preserves context efficiently
- Handles long conversations

**Cons:**
- Summarization cost
- Summary quality varies

**When to use:** Extended conversations where context is important

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(),
    max_token_limit=500,  # Summarize if > 500 tokens
)
# Automatically summarizes when token limit exceeded
```

#### 4. Custom Memory
```python
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationMemory

class CustomMemory(ConversationMemory):
    def load_memory_variables(self, inputs):
        # Your custom logic
        return {"chat_history": self.get_formatted_history()}
    
    def save_context(self, inputs, outputs):
        # Custom storage logic (database, file, etc)
        pass
```

### Memory Flow in Conversations

```
User 1: "Hi, my name is John"
  ↓ (Save to memory)
Memory: [("Hi, my name is John", "Nice to meet you, John")]
  ↓
User 2: "What's my name?"
  ↓ (Retrieve memory + inject into prompt)
Prompt: "History: John introduced himself. User asks: What's my name?"
  ↓
LLM Response: "Your name is John"
  ↓
Save: [(original exchange), ("What's my name?", "Your name is John")]
```

### Interview Definition
> "Memory persists conversational context. Buffer memory stores all messages; window memory keeps recent messages; summarized memory condenses old messages to save tokens."

---

## 🔑 KEY POINTS FOR REVISION

### Core Understanding
- ✅ **LangChain is a framework** that abstracts LLM complexity
- ✅ **Runnables are composable** (pipe operator `|` chains them)
- ✅ **Chains are Runnables + Runnables** (building blocks)
- ✅ **Agents think and decide** which tools to use
- ✅ **RAG grounds LLMs** in external knowledge
- ✅ **Output Parsers ensure** data validity
- ✅ **Memory provides** conversation context
- ✅ **Tools are LLM-callable** functions
- ✅ **Prompts are templates** with variable injection
- ✅ **Chat Models vs LLMs** — Chat is the modern standard

### Common Interview Traps
1. **"LLMs remember conversations"** → ❌ Wrong, they're stateless. Use memory.
2. **"Agents always find the best solution"** → ❌ They can hallucinate. Validate outputs.
3. **"More context in prompts = better answers"** → ⚠️ Token limits and relevance matter more.
4. **"RAG eliminates hallucinations"** → ⚠️ RAG reduces hallucinations but doesn't eliminate them.
5. **"Tools run automatically when LLM mentions them"** → ❌ Tool execution is explicit.

### Performance Considerations
- **Token Optimization**: Compress history with summarization
- **Latency**: Use `stream()` for real-time responses
- **Cost**: Monitor API calls, batch when possible
- **Accuracy**: Validate with output parsers
- **Scale**: Use vector DBs for large document sets

### Architecture Pattern (Remember This)
```
INPUT
  ↓
PROMPT TEMPLATE (Format user query)
  ↓
LLM/CHAT MODEL (Get response)
  ↓
OUTPUT PARSER (Validate & structure)
  ↓
CHAIN/AGENT (Navigate complex logic)
  ↓
TOOLS (External knowledge/actions)
  ↓
MEMORY (Context persistence)
  ↓
OUTPUT
```

---

## ❓ INTERVIEW QUESTIONS (With Answers)

### BEGINNER LEVEL

**Q1. What is LangChain and why do we need it?**
> LangChain is a framework that simplifies building LLM applications. It provides abstractions over different LLM providers (OpenAI, Google, Hugging Face), handling prompt management, tool integration, memory, and output parsing. Without it, you'd write custom API code for each provider.

**Q2. What's the difference between an LLM and a Chat Model?**
> LLMs take text-in and return text-out (stateless). Chat Models take structured messages (with roles like system/user/assistant) and are optimized for conversations. Chat Models are instruction-tuned and the modern standard for applications.

**Q3. What is a Runnable?**
> A Runnable is LangChain's standard interface for composable units of work. Any component (prompt, model, tool) that takes input and produces output is a Runnable. They support `.invoke()`, `.stream()`, `.batch()` and can be chained with the `|` operator.

**Q4. Explain this code: `chain = prompt | model | parser`**
> This creates a sequential chain where:
1. `prompt` formats the input
2. `model` generates a response to the formatted prompt
3. `parser` converts the raw response into structured data
> The pipe operator `|` implements a chain composition pattern.

**Q5. What's the difference between a Chain and an Agent?**
> **Chains** follow predetermined paths using pipes (`|`). They're predictable but inflexible.
> **Agents** use LLM reasoning to dynamically choose actions. They think, plan, and adapt—making them flexible but less predictable.

---

### INTERMEDIATE LEVEL

**Q6. How does Tool Binding and Tool Calling work?**
> **Tool Binding**: Registers tools with an LLM so it knows what's available via JSON schema.
```python
llm_with_tools = llm.bind_tools([search_tool, calculator_tool])
```
> **Tool Calling**: LLM decides to use a tool and suggests parameters (doesn't execute).
> **Tool Execution**: Actual function runs with suggested parameters.
> **Result Injection**: LLM sees the tool output and determines next steps.

**Q7. Explain the RAG pipeline step-by-step.**
> 1. **Load**: Documents loaded into standardized format
> 2. **Split**: Long texts chunked into manageable pieces
> 3. **Embed**: Chunks converted to vectors via embedding model
> 4. **Store**: Vectors stored in DB with metadata
> 5. **Query**: User query converted to vector
> 6. **Retrieve**: Similar vectors found via k-NN
> 7. **Context**: Retrieved documents injected into prompt
> 8. **Generate**: LLM uses context for grounded answer

**Q8. Why do we use Output Parsers like Pydantic?**
> Raw LLM responses are unstructured text. Pydantic parsers:
> - ✅ Enforce type safety (`sentiment: str`, `rating: int`)
> - ✅ Validate data (constraints, ranges)
> - ✅ Convert to Python objects (easy to use)
> - ✅ Fail gracefully if format is wrong

**Q9. What's the difference between ConversationBufferMemory and ConversationBufferWindowMemory?**
> **BufferMemory**: Stores all messages, complete context, but grows unbounded.
> **WindowMemory**: Keeps only last K messages, bounded size, but loses early context.
> **Choose BufferMemory for short conversations, WindowMemory for long chats.**

**Q10. Explain prompt templates vs. f-strings.**
```python
# f-string approach:
response = llm(f"Write about {topic}")

# Prompt template approach:
template = PromptTemplate(
    template="Write about {topic}",
    input_variables=["topic"]
)
```
> **Advantages of templates**:
> - Part of LangChain ecosystem (compose with other components)
> - Validation built-in
> - Support partial formatting
> - Support message placeholders for conversation history
> - Better for complex prompts

---

### ADVANCED LEVEL

**Q11. Design an RAG system for a large document library (100MB+). What are the challenges?**
> **Challenges**:
> 1. **Chunking Strategy**: Balance size vs. context. Too small = fragmented; too large = token waste
> 2. **Embedding Cost**: 100MB → millions of vectors = expensive embedding API calls
> 3. **Vector DB Scale**: Need distributed DB (Pinecone, Weaviate) not in-memory Chroma
> 4. **Retrieval Quality**: k-NN may miss semantically relevant docs, use reranking
> 5. **Update Frequency**: Incremental updates vs. full re-embedding
> 6. **Latency**: Multi-step pipeline (embed → retrieve → LLM) needs optimization

> **Solution**:
```python
# 1. Use semantic chunking
from langchain.text_splitters import SemanticChunker
splitter = SemanticChunker(embeddings, separators=[".", " "])

# 2. Batch embed to reduce cost
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(batch_size=100)

# 3. Use production vector DB
from langchain_community.vectorstores import Pinecone
vector_store = Pinecone.from_documents(
    docs, embeddings, index_name="docs-index"
)

# 4. Add reranking
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm, prompt)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)

# 5. Stream responses for lower latency
for chunk in chain.stream({"query": user_query}):
    print(chunk, end="", flush=True)
```

**Q12. How would you implement persistent memory for a conversational agent?**
> **Requirements**: Survive across sessions, scalable, queryable.

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
import json
from pathlib import Path

class PersistentMemory:
    def __init__(self, file_path="memory.json"):
        self.file_path = file_path
        self.memory = self._load()
    
    def _load(self):
        if Path(self.file_path).exists():
            with open(self.file_path) as f:
                return json.load(f)
        return []
    
    def save(self, interaction):
        self.memory.append(interaction)
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f)
    
    def get_context(self, k=5):
        """Return last k interactions"""
        return self.memory[-k:] if self.memory else []

# Or use a database:
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

class DatabaseMemory:
    def __init__(self, db_url="sqlite:///memory.db"):
        self.engine = create_engine(db_url)
    
    def save_interaction(self, user_id, role, content):
        # Store in DB with timestamp
        pass
    
    def get_conversation_history(self, user_id, limit=50):
        # Query from DB
        pass
```

**Q13. Compare Sequential Chain vs. Parallel Chain vs. Conditional Chain. When use each?**

| Chain Type | Use When | Example | Flow |
|-----------|----------|---------|------|
| **Sequential** | Tasks have dependencies | Summarize → Translate | A → B → C |
| **Parallel** | Independent tasks | Generate title & summary simultaneously | A → Output / B → Output |
| **Conditional** | Decision logic needed | Check sentiment, route to different chains | If happy → escalate, else → resolve |

**Q14. What are common failure modes in RAG systems?**
> 1. **Poor Chunking**: Semantically incomplete chunks misses answers
> 2. **Retrieval Failure**: k-NN doesn't find relevant docs (use reranking)
> 3. **Context Injection Failure**: Retrieved docs don't answer question
> 4. **LLM Hallucination**: LLM ignores context and makes up answers
> 5. **Token Overflow**: Context too large for model's context window
> 6. **Stale Documents**: Knowledge base outdated

**Q15. Design an intelligent Agent for customer support. What tools and design patterns?**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for articles"""
    # Search implementation
    return knowledge_base.search(query)

@tool
def check_order_status(order_id: str) -> dict:
    """Check customer order status"""
    return database.get_order(order_id)

@tool
def create_ticket(issue: str, priority: str) -> int:
    """Create support ticket for complex issues"""
    return ticketing_system.create(issue, priority)

@tool
def process_refund(order_id: str, reason: str) -> bool:
    """Process refund for order"""
    return payment_system.refund(order_id, reason)

tools = [search_knowledge_base, check_order_status, create_ticket, process_refund]

llm = ChatOpenAI(model="gpt-4")

system_prompt = """You are a helpful customer support agent. 
Your goal is to resolve customer issues efficiently.

First, try to solve using the knowledge base.
If complex, escalate with a support ticket.
Always ask clarifying questions before taking action.
Be empathetic and professional."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Usage:
response = agent_executor.invoke({
    "input": "My order #12345 hasn't arrived",
    "chat_history": [],
    "agent_scratchpad": ""
})
```

---

## 🚀 ADVANCED INSIGHTS (Master It)

### Performance Optimization Strategies

**1. Token Optimization**
```python
# Problem: Long context wastes tokens
# Solution: Compress with summarization

from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),  # Cheap model for summarization
    max_token_limit=500,  # Summarize when > 500 tokens
)
```

**2. Lazy Loading**
```python
# Problem: Loading entire document dataset is slow
# Solution: Load documents lazily (on demand)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("large_file.txt")
# loader.load() - loads immediately
docs = loader.lazy_load()  # Loads documents one-by-one when iterated
```

**3. Streaming for Latency**
```python
# Problem: Waiting for full LLM response adds latency
# Solution: Stream tokens as they arrive

for chunk in chain.stream({"query": "Explain quantum computing"}):
    print(chunk, end="", flush=True)  # Real-time output
```

**4. Parallel Execution**
```python
# Problem: Sequential tool calls are slow
# Solution: Run tools in parallel

from langchain_core.runnables import RunnableParallel

parallel_tasks = RunnableParallel(
    weather=weather_tool_chain,
    traffic=traffic_tool_chain,
    news=news_tool_chain
)

results = parallel_tasks.invoke({"location": "NYC"})
# All three run simultaneously, not sequentially
```

**5. Batch Processing**
```python
# Problem: Single query at a time is inefficient
# Solution: Batch process multiple queries

queries = ["What is AI?", "Define ML", "Explain NLP"]

results = chain.batch([{"topic": q} for q in queries])
# Processes all three efficiently
```

### Scaling Considerations

| Scale Level | Challenge | Solution |
|-------------|-----------|----------|
| **100 docs** | In-memory Chroma is fine | Semantic search sufficient |
| **10K docs** | Memory concerns | Use Pinecone/Weaviate, add reranking |
| **100K+ docs** | Latency & cost | Distributed vector DB, sharding, hybrid search |
| **Real-time updates** | Index staleness | Incremental embedding, change tracking |
| **Multi-user** | Concurrency | Cache popular queries, async processing |

### Industry Usage Patterns

**1. E-Commerce Chatbots**
- RAG for product knowledge
- Tools for inventory, orders, payments
- Memory for user preferences

**2. Customer Support AI**
- RAG for knowledge base
- Tools for ticket creation, refunds
- Memory for conversation continuity

**3. Code Assistants**
- RAG for documentation, code examples
- Tools for file operations, testing
- Memory for context within editor

**4. Research Assistants**
- RAG for academic papers, data
- Tools for web search, data analysis
- Memory for research progress

### When NOT to Use LangChain
- ❌ Simple API calls (just use requests library)
- ❌ Real-time applications needing sub-100ms latency
- ❌ Cost-sensitive applications (API calls add latency/expense)
- ❌ Applications not using LLMs
- ❌ Extremely custom workflows (framework may be overkill)

### Common Anti-Patterns to Avoid

**1. RAG without reranking**
```python
# ❌ Bad: Just return top-K
docs = vector_store.similarity_search(query, k=5)

# ✅ Good: Rerank results for relevance
from langchain.retrievers import ContextualCompressionRetriever
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(k=10)
)
docs = retriever.get_relevant_documents(query)  # Returns top-5 after reranking
```

**2. Trusting LLM without validation**
```python
# ❌ Bad
result = llm.invoke(prompt)
save_to_database(result)  # What if hallucination?

# ✅ Good
result = structured_llm.invoke(prompt)  # Pydantic enforces schema
if validate(result):  # Check constraints
    save_to_database(result)
```

**3. No fallbacks**
```python
# ❌ Bad: Fails if one tool fails
result = tool1_chain | tool2_chain | tool3_chain

# ✅ Good: Define fallbacks
result = (tool1_chain | tool2_chain) | default_chain
# or
try:
    result = tool1_chain.invoke(input)
except Exception:
    result = fallback_chain.invoke(input)
```

**4. Overwhelming context**
```python
# ❌ Bad: Dump everything in prompt
prompt = f"""Context: {all_documents} \n Query: {user_query}"""

# ✅ Good: Carefully select relevant context
relevant_docs = retriever.get_relevant_documents(user_query)
prompt = f"""Context: {relevant_docs[:3]} \n Query: {user_query}"""
```

---

## 📚 QUICK REVISION SUMMARY (5-7 Lines)

**LangChain** is a framework that abstracts LLM complexity into composable **Runnables** (linked with `|`). Core components: **Prompts** (templates), **Models** (LLMs/Chat), **Chains** (sequential/parallel/conditional), **Tools** (LLM-callable functions), **Agents** (reasoning + planning), **RAG** (external knowledge via retrieval), **Memory** (conversation persistence), **Parsers** (structured output). Typical flow: Prompt → LLM → Parser → Chain → Agent → Tools → Output. Master the distinction between LLMs (stateless text) and Chat Models (stateful conversations), understand how RAG grounds responses with documents, use Pydantic for output validation, and implement memory for context management.

---

## 🎯 QUICK PROJECT STRUCTURE REFERENCE

```
LangChainNotes/
├── ChainS/
│   ├── SimpleChain.py (PromptTemplate → LLM → Parser)
│   ├── SequentialChan.py (A → B → C pipelines)
│   ├── ParllelChains.py (Parallel execution)
│   └── ConditionalChain.py (If/else logic)
├── LLMs/
│   ├── chatmodels.py (Chat vs LLM comparison)
│   ├── googleModel.py (Google Generative AI)
│   ├── llm_demo.py (LLM usage examples)
│   └── promptsPlaceHolder.py (Dynamic prompts)
├── Tools/
│   ├── duckDuck.py (DuckDuckGo search tool)
│   └── [Custom tools implementation]
├── Rag/
│   ├── DocumentLoader/ (load .txt, .pdf, .csv)
│   ├── TextSplitter/ (chunking strategies)
│   ├── VectorStore/ (embeddings storage)
│   └── Retrivers/ (similarity search)
├── StructureOutput/
│   ├── withStructured_pydantic.py (Pydantic schema)
│   ├── with_structure_output_json.py (JSON schema)
│   └── JsonParser.py (JSON output parsing)
├── Runnable/
│   └── [Runnable composition examples]
└── [Documentation & notes]
```

---

## 🔗 FILE-BY-FILE QUICK REFERENCE

### SimpleChain.py
- Demonstrates basic chain: PromptTemplate → GoogleGenerativeAI → StrOutputParser
- Uses `.invoke()` to execute
- Shows `.get_graph().print_ascii()` for visualization

### DuckDuck.py
- Shows DuckDuckGoSearchRun() for web search
- Example: `search.invoke("Obama's first name?")`

### withStructured_pydantic.py
- Creates Pydantic model for Review extraction
- Uses `model.with_structured_output(Review)` for parsing
- Example: Parse product review into structured format

### loadingText.py
- Uses TextLoader to load .txt files
- Returns Document objects with page_content and metadata

---

**Created**: 2026-04-10
**Updated**: Based on your entire LangChainNotes project
**Level**: Interview Ready (Beginner → Advanced)
**Topics Covered**: 10 major areas
**Questions**: 15 detailed interview questions with answers
