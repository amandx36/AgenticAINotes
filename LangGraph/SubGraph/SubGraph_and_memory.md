# Subgraphs and Memory

## Overview
This note explores the role of subgraphs in LangGraph and how memory interacts with nested workflows.

![alt text](image.png)

## Why Subgraphs?
Subgraphs let you reuse workflow logic inside a larger graph. They can be referenced by a node or embedded as a node itself.

![alt text](image-1.png)

## Subgraph Types
1. **Graph reference node**: a node that invokes another graph by reference.
2. **Embedded subgraph node**: a node that contains a subgraph as its internal logic.

![alt text](image-2.png)

## Important Links
- LangChain Subgraphs docs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs?utm_source=chatgpt.com
- Example stream subgraph outputs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs?utm_source=chatgpt.com#stream-subgraph-outputs

## Memory and LLMs
LLMs do not have intrinsic memory. Memory must be built externally and supplied as context or retrieved state.

## Context Window
![alt text](image-3.png)

### In-Context Learning
In-context learning uses prompt examples or prior dialogue to teach the model within the same request.

![alt text](image-5.png)

## Solution Principle
![alt text](image-4.png)

## Short-Term Memory
![alt text](image-6.png)

### Problems with Short-Term Memory
1. Short-term memory is fragile and can break if the context grows too large.
2. The context window is limited by the model's maximum token capacity.

### Recommended Solutions
- Persist critical state externally in a database.
- Trim or summarize context as needed.

![alt text](image-7.png)

## Thread Scope
Short-term memory is usually thread-scoped.

### Limitations
- Loss of user continuity across conversations
- Learning does not compound over time
- Cross-thread reasoning is impossible

## Long-Term Memory
![alt text](image-8.png)

### Types of Long-Term Memory
![alt text](image-9.png)

## How Long-Term Memory Works
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
![alt text](image-13.png)

## Libraries and Research
- Popular memory library: LangMem
- Other related systems: Mem, Super-memory
- Research: Google papers on memory for LLMs such as TITANs and MemRA
