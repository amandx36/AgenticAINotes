# RAG and Human-In-The-Loop

## 1. RAG (Retrieval-Augmented Generation)
RAG solves knowledge cutoff and private data access problems by grounding LLM outputs with retrieved external documents.

### Why RAG Matters
LLM hallucinations occur when models invent facts. RAG reduces this by feeding the model evidence from a document store or knowledge base.

### Core RAG Flow
1. User query arrives.
2. Retriever searches the knowledge base for relevant text.
3. Top candidate chunks are ranked and filtered.
4. Context is appended to the prompt.
5. The LLM generates a grounded response.

### Screenshot Explanation
![alt text](image.png)
This image likely shows the RAG workflow with query, retrieval, and generation stages.

### Production Insight
Implement RAG as a separate tool or microservice so retrieval can be monitored, reranked, and audited independently from generation.

## 2. HITL (Human-in-the-Loop)
Human-in-the-loop is a workflow safety pattern in which a human reviews, approves, or corrects model outputs at critical checkpoints.

### Why HITL Matters
HITL improves accountability, accuracy, and safety in agentic systems. It is essential for high-risk decisions, compliance, and ambiguous operations.

### Typical HITL Points
- before performing external actions
- before final answer delivery
- when confidence is low
- for regulatory or ethical review

### Interview Angle
An interviewer may ask: “Where would you place HITL checkpoints in a retrieval or agentic workflow?” or “How does HITL impact latency and user experience?”

### Follow-Up
- How does HITL differ from automated validation?
- What metrics would show HITL effectiveness?
- How do you keep HITL from becoming a bottleneck?
