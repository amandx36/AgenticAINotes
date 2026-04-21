# 🧪 RAG Deep Dive - Complete Implementation Guide

## Table of Contents
1. [RAG Architecture](#rag-architecture)
2. [Document Loading Strategies](#document-loading-strategies)
3. [Text Splitting Techniques](#text-splitting-techniques)
4. [Vector Databases](#vector-databases)
5. [Retrieval Optimization](#retrieval-optimization)
6. [End-to-End RAG Example](#end-to-end-rag-example)
7. [RAG Interview Questions](#rag-interview-questions)

---

## RAG Architecture

### What RAG Does
RAG = **Retrieval-Augmented Generation**
- Retrieves relevant documents from knowledge base
- Augments (adds) to LLM prompt as context
- Generates response grounded in real data

### Why RAG is Critical
```
Problem: LLM has knowledge cutoff, no access to private data, hallucination
Solution: RAG bridges this gap

LLM Knowledge + RAG External Knowledge = Accurate, Current Answers
```

### RAG vs Regular LLM

| Aspect | LLM Alone | LLM + RAG |
|--------|-----------|----------|
| **Knowledge** | Training data only (cutoff) | + Live/private documents |
| **Hallucination** | Likely for unknown topics | Reduced (grounded) |
| **Freshness** | Stale (retrain needed) | Current |
| **Cost** | Lower latency | Adds retrieval overhead |
| **Accuracy** | ±~70% | ±~85% (varies) |

---

## Document Loading Strategies

### 1. TextLoader (Plain Text)
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")
documents = loader.load()

# Output:
# [Document(
#    page_content="Cricket is a bat-and-ball sport...",
#    metadata={"source": "cricket.txt"}
# )]

print(documents[0].page_content)  # Access text
print(documents[0].metadata)      # Access metadata
```

**When to use:** Knowledge base TXT files, logs, plain documents
**Limitation:** No structure recognition

---

### 2. PDFLoader (PDF Documents)
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("research_paper.pdf")
documents = loader.load()

# Loads page-by-page
for doc in documents:
    print(doc.metadata["page"])  # Page number
```

**When to use:** Research papers, reports, manuals
**Limitation:** Poor with images/complex layouts. For image-heavy PDFs, use specialized loaders.

---

### 3. DirectoryLoader (Multiple Files)
```python
from langchain_community.document_loaders import DirectoryLoader

# Load all TXT files in folder
loader = DirectoryLoader("./documents", glob="**/*.txt")
documents = loader.load()

# Load with specific loader
loader = DirectoryLoader(
    "./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()
```

**When to use:** Batch loading entire knowledge bases
**Lazy loading for memory efficiency:**
```python
# Load one-by-one instead of all at once
docs = loader.load_and_split()
```

---

### 4. WebBaseLoader (Web Pages)
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

**When to use:** Documentation sites, blogs, news articles
**Advanced:** Load multiple URLs
```python
urls = ["url1.com", "url2.com", "url3.com"]
loaders = [WebBaseLoader(url) for url in urls]
documents = []
for loader in loaders:
    documents.extend(loader.load())
```

---

### 5. CSVLoader (Structured Data)
```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("products.csv")
documents = loader.load()

# Converts rows to text documents
# Example: "Name: Apple Laptop, Price: $999, Stock: 45"
```

**When to use:** Database exports, spreadsheets, tabular data
**Limitation:** Works best with reasonably wide format (not too many columns)

---

### 6. Custom Loaders
```python
from langchain_core.documents import Document

class CustomJSONLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        import json
        with open(self.file_path) as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                page_content=f"Title: {item['title']}\n{item['content']}",
                metadata={"source": self.file_path, "id": item["id"]}
            )
            documents.append(doc)
        
        return documents

loader = CustomJSONLoader("data.json")
docs = loader.load()
```

---

### Document Structure After Loading
```python
Document {
    page_content: "The actual text or content",
    metadata: {
        "source": "filename.pdf",
        "page": 1,           # Page number (for PDFs)
        "chunk_id": 0,       # Chunk index (if split)
        "custom_field": "..."  # Your custom metadata
    }
}
```

---

## Text Splitting Techniques

### 1. Length-Based (Simple, but dumb)
```python
from langchain.text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,      # Max 1000 chars per chunk
    chunk_overlap=200     # 200 char overlap between chunks
)

chunks = splitter.split_text(my_large_text)
# Result: ["chunk1...", "chunk2...", "chunk3..."]
```

**Flow:**
```
Text: "The quick brown fox jumps over the lazy dog. The dog was sleeping."
  ↓ Split every 1000 chars
Chunk 1: "The quick brown fox jumps over [200 overlap chars]"
Chunk 2: "[200 overlap chars] the lazy dog. The dog was sleeping."
```

**Problem:** May split mid-sentence
**When to use:** Simple texts, demos

---

### 2. Recursive Character Split (Smart)
```python
from langchain.text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]  # Priority order
)

chunks = splitter.split_documents(documents)
```

**Flow (Priority-based):**
```
Text: "Para 1\n\nPara 2\n\nPara 3"
Step 1: Try splitting by "\n\n" (paragraphs)
  → ["Para 1", "Para 2", "Para 3"]
If still > 500 chars:
Step 2: Try splitting by "\n" (lines)
If still > 500 chars:
Step 3: Try splitting by " " (words)
If still > 500 chars:
Step 4: Split by "" (characters)
```

**Advantage:** Preserves structure
**When to use:** Documents with structure (essays, articles)

---

### 3. Document Structure-Based (Respect formatting)
```python
from langchain.text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

splitter.split_text(markdown_text)
```

**Example:**
```
Input:
# Main Title
Some content...

## Subtitle
More content...

### Sub-subtitle
Even more...

Output:
Document 1: "Some content..." (metadata: Header 1="Main Title", Header 2="")
Document 2: "More content..." (metadata: Header 1="Main Title", Header 2="Subtitle")
Document 3: "Even more..." (metadata: Header 1="Main Title", Header 2="Subtitle", Header 3="Sub-subtitle")
```

**When to use:** Markdown docs, structured documentation

---

### 4. Semantic Split (Preserve Meaning!)
```python
from langchain.text_splitters import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(
    embeddings=embeddings,
    separators=[".", " "],
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # Split when semantic similarity drops >95%
)

chunks = splitter.split_text(text)
```

**How it works:**
```
Sentence 1: "I love programming"
Sentence 2: "coding is fun"        → Similar meaning (keep together)
Sentence 3: "The weather is rainy"  → Different meaning (START NEW CHUNK)
```

**Advantage:** Maintains semantic coherence
**Cost:** Embedding changes needed (more API calls)
**When to use:** Technical docs, research papers where meaning is critical

---

### 5. Language-Specific Splitting
```python
from langchain.text_splitters import Language, RecursiveCharacterTextSplitter

# Python code
code = """
def hello():
    print("world")
    
class MyClass:
    pass
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20
)

chunks = python_splitter.split_text(code)
# Respects Python structure (functions, classes, indentation)
```

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, SQL, Java, C++, C#, Markdown, LaTeX, Solidity

---

### Text Splitting Best Practices

| Aspect | Best Practice |
|--------|----------------|
| **Chunk Size** | 500-1000 tokens (not characters!) |
| **Overlap** | 10-20% of chunk size |
| **Splitting Method** | Recursive (respects structure) |
| **Semantics** | Use if quality > performance needs |
| **Metadata** | Add source, chunk_id, page_num |
| **Testing** | Verify chunks make sense |

---

## Vector Databases

### 1. Chroma (In-Memory, Easy)
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Create and store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_knowledge_base"
)

# Retrieve
results = vector_store.similarity_search("What is AI?", k=3)
```

**Pros:** Easy setup, fast, good for dev
**Cons:** In-memory, not scalable for production
**Use:** Prototypes, small datasets (<10K docs)

---

### 2. Pinecone (Cloud Vector DB)
```python
from langchain_community.vectorstores import Pinecone
import pinecone

# Setup Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
index_name = "langchain-docs"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

# Store documents
vector_store = Pinecone.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    index_name=index_name
)

# Retrieve
results = vector_store.similarity_search("How does RAG work?", k=5)
```

**Pros:** Fully managed, scalable, fast, metadata filtering
**Cons:** Cloud dependency, cost
**Use:** Production, large-scale applications

---

### 3. Weaviate (Open-Source + Cloud)
```python
from langchain_community.vectorstores import Weaviate
import weaviate

# Connect to Weaviate instance
client = weaviate.Client("http://localhost:8080")

# Store documents
vector_store = Weaviate.from_documents(
    documents=chunks,
    client=client,
    by_text=False,
    embedding=OpenAIEmbeddings(),
)

# Retrieve with filtering
results = vector_store.similarity_search(
    query="Advanced AI techniques",
    where_filter={
        "path": ["source"],
        "operator": "Equal",
        "valueString": "research_paper.pdf"
    },
    k=5
)
```

**Pros:** Open-source option, hybrid search, metadata filtering
**Cons:** More complex setup
**Use:** Enterprise apps, on-premises requirements

---

### 4. Milvus (Open-Source, Distributed)
```python
from langchain_community.vectorstores import Milvus
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host="127.0.0.1", port=19530)

# Store documents
vector_store = Milvus.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="langchain_docs"
)

# Retrieve
results = vector_store.similarity_search("ML algorithms", k=5)
```

**Pros:** Open-source, distributed, scalable
**Cons:** Complex deployment
**Use:** Large-scale, distributed systems

---

### Vector DB Comparison

| DB | Type | Scalability | Cost | Setup | Metadata |
|----|------|-------------|------|-------|----------|
| **Chroma** | In-memory | Small | Free | Easy | Limited |
| **Pinecone** | Cloud | Large | Paid | Easy | Excellent |
| **Weaviate** | Cloud + OSS | Large | Free/Paid | Medium | Excellent |
| **Milvus** | Distributed | Huge | Free | Hard | Good |

---

## Retrieval Optimization

### Problem: Basic k-NN Retrieval Misses Relevant Docs

```python
# ❌ Bad: Returns irrelevant results
query = "How to start a side project?"
top_3 = vector_store.similarity_search(query, k=3)

# Might return docs about "side effects" not "side project"!
```

### Solution 1: Contextual Compression Retrieval

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

# Extract relevant parts using LLM
compressor = LLMChainExtractor.from_llm(llm)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(k=10)
)

# Reranks: retrieves 10, filters to 3-5 most relevant
results = retriever.get_relevant_documents("Starting a side hustle")
```

**Flow:**
```
Query → Retrieve 10 similar docs
  ↓
LLM filters: "Is this relevant?"
  ↓
Keep 3-5 most relevant
```

**Good for:** Medium datasets, quality > speed

---

### Solution 2: Hybrid Search (Vector + Keyword)

```python
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Vector retriever
vector_retriever = vector_store.as_retriever(k=5)

# Keyword retriever (BM25)
keyword_retriever = BM25Retriever.from_documents(chunks)

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.6, 0.4]  # 60% vector, 40% keyword
)

results = ensemble_retriever.get_relevant_documents("Python async")
```

**Why both?**
- Vector search: Semantic understanding ("async" vs "concurrent")
- Keyword search: Exact matches ("Python" literal)
- Combined: Best of both worlds

---

### Solution 3: Multi-Query Retrieval

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=ChatOpenAI()
)

# LLM generates multiple queries automatically:
# "How to build APIs?" →
# 1. "API development guide"
# 2. "REST API best practices"
# 3. "Building web services"
# Retrieves docs for all variants, deduplicates, returns best

results = retriever.get_relevant_documents("How to build APIs?")
```

---

### Solution 4: Parent Document Retriever

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Strategy: Store parent (full section) + child (small chunks)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma.from_documents(
    documents=child_chunks,  # Search in small chunks
    embedding=embeddings
)

store = InMemoryStore()  # Store full parent docs

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Retrieves by small chunk, returns full parent section
results = retriever.get_relevant_documents("ML algorithms")
```

**Benefit:** Speed of small chunks + context of large docs

---

## End-to-End RAG Example

### Complete RAG Pipeline
```python
from langchain_community.document_loaders import PDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load
loader = PDFLoader("research_paper.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2: Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Embed & Store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="research"
)
print("Stored in vector DB")

# Step 4: Retrieve
query = "What are the main findings?"
relevant_docs = vector_store.similarity_search(query, k=5)
context = "\n".join([doc.page_content for doc in relevant_docs])

# Step 5: Generate
llm = ChatOpenAI(model="gpt-4")

template = """Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"context": context, "query": query})
print(f"Answer: {answer}")
```

---

## RAG Interview Questions

**Q1: What is RAG and why is it important?**
> RAG = Retrieval-Augmented Generation. It retrieves external documents relevant to a query and injects them as context into the LLM prompt. Important because it grounds LLM responses in real data, reducing hallucinations and providing current, accurate information.

**Q2: How does chunking affect RAG quality?**
> Too small chunks: Fragmented context, missing meaning.
> Too large chunks: Token waste, slower retrieval.
> Optimal: ~500-1000 tokens with 10-20% overlap. Use semantic splitting for coherence.

**Q3: Compare in-memory (Chroma) vs cloud (Pinecone) vector DBs.**
> **Chroma**: Easy, fast, dev-friendly, but limited to small datasets.
> **Pinecone**: Scalable, production-ready, managed, but cloud dependency and cost.
> Choose Chroma for prototypes, Pinecone for production.

**Q4: How do you prevent RAG from returning irrelevant documents?**
> 1. Use higher quality chunks (semantic splitting)
> 2. Add reranking (ContextualCompressionRetriever)
> 3. Use hybrid search (vector + keyword)
> 4. Filter by metadata
> 5. Use multi-query to catch variations

**Q5: Explain the difference between similarity_search and hybrid search.**
> **Similarity search**: Uses vector embeddings, finds semantic matches but misses exact keywords.
> **Hybrid search**: Combines vector (semantic) + keyword (BM25) searches, better overall quality.

**Q6: Design a RAG system for 100K documents. What challenges?**
> Challenges: Chunking strategy, embedding cost, vector DB scale, retrieval latency, update frequency, reranking.
> Solution: Semantic chunking, batch embedding, cloud vector DB (Pinecone), reranking with LLM, incremental updates, async processing.

**Q7: How do you handle document updates in a RAG system?**
> Options:
> 1. Full re-embed: Simple but expensive
> 2. Incremental: Only new/changed docs
> 3. Versioning: Keep history, update metadata
> 4. Hybrid: Batch updates during off-peak

**Q8: Why is overlap important in chunking?**
> Overlap prevents context loss at chunk boundaries.
> Without overlap: "...the algorithm works because it..." → "...of the new method" (disconnected)
> With overlap: Context flows smoothly between chunks.

**Q9: What's the difference between semantic and length-based chunking?**
> **Length-based**: Fixed size chunks, simple, may split mid-sentence.
> **Semantic**: Preserves meaning, more intelligent, costs more (embeddings).
> Length-based for speed, semantic for quality.

**Q10: How do you measure RAG quality?**
> Metrics:
> 1. **Retrieval quality**: Precision (relevance) and recall (coverage)
> 2. **Generation quality**: BLEU, ROUGE (comparing to ground truth)
> 3. **Hallucination rate**: % of unfounded claims
> 4. **User satisfaction**: Feedback scores
> 5. **Latency**: End-to-end response time

---

**Takeaway:** RAG is the bridge between static LLM knowledge and dynamic, grounded applications. Master chunking, embedding, and retrieval strategies for production systems.
