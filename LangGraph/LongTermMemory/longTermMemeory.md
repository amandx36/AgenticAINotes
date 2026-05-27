# Long-Term Memory

## What This File Explains
This note explains how to build a long-term memory store for LangGraph using a vector-based persistence layer.

## Memory Store Architecture
A long-term memory store typically uses a `BaseStore` abstraction for:
- create memory items
- search existing memory
- update memories
- delete memories

![alt text](image.png)

## Namespace-Based Organization
Use namespaces to organize memories by user, session, or application domain.
A namespace separates memory entries and avoids collisions.

### Put Method
A `put` operation usually requires:
- `namespace`
- `key`
- `value`

![alt text](image-1.png)

## Semantic Search for Retrieval
To retrieve the right memory item, use semantic search instead of exact key matching.
This relies on embedding models that represent text meaning in vector space.

## Using Existing Memory
When a query arrives, check whether related memory already exists before creating a new entry.

![alt text](image-2.png)

## Memory Creation Example
The following screenshot likely shows code for inserting a new memory into a vector store.

![alt text](image-3.png)

## Duplicate Memory Handling
To avoid duplicates:
1. Send the new user message.
2. Compare it against existing memories.
3. Return a list of matches with boolean flags indicating whether each candidate is already stored.
4. Insert only memories marked as `False`.

## Combined Workflow
The memory workflow should support both update and create paths.

![alt text](image-4.png)

## Production Insight
Long-term memory should be deduplicated and searchable by semantic similarity. Use namespaces and vector stores for scalability, and avoid storing raw unstructured data when only vector references are needed.
