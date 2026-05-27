# Short-Term Memory Implementation

## Purpose
This note explains short-term memory handling in LangGraph workflows, focusing on checkpointing, threads, and context overflow management.

## Core Mechanisms
Short-term memory often uses:
1. Checkpointers
2. Threads

These mechanisms keep workflow state local to each user session and allow resumption after interruptions.

## Context Overflow Problem
Context overflow occurs when the combined prompt, system instructions, and generated output exceed the LLM token limit.

## Overcome Methods

### 1. Trimming
Remove unnecessary old messages when the token count is too large.
- Set a maximum token budget.
- Trim the oldest messages or the least relevant context.

![alt text](image.png)

LangGraph may provide a `trim_message` utility to support this behavior.

![alt text](image-1.png)

### Limitations of Trimming
- Old messages are not deleted permanently.
- Only the working context is shortened.

### 2. Summarization
Summarize older conversation history and keep only the compressed summary plus recent messages.

![alt text](image-2.png)

## Production Insight
Use summarization to preserve semantic context without consuming unnecessary tokens. Use trimming only when summaries are unavailable or when strict budget control is required.

## Example Imports
```python
from langchain.messages import RemoveMessages
```
