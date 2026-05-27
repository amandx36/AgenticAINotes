# LangGraph Persistence
Persistence saves and restores the state of a workflow over time.

## Why Persistence Matters
Persistence provides:
- fault tolerance
- checkpointing
- resume after failure
- long-running workflow continuity
- human-in-the-loop pauses

## Key Concepts
### State
The shared structured data passed between nodes in a graph.
Example:
```python
state = {
    "query": "Hire backend engineer",
    "plan": [],
    "candidates": [],
    "selected": None
}
```

### Checkpoint
A saved snapshot of workflow state at a defined step.

### Thread
A unique workflow session identifier.

### Checkpointer
The storage component that saves and restores checkpoints.

## Benefits
- Short-term memory support
- Fault tolerance
- HITL support
- Time travel debugging

## Persistence vs Memory
- Persistence saves workflow state and progress.
- Long-term memory stores user history, preferences, and knowledge.

## Implementation Notes
- Use a checkpointer only at the parent graph.
- Avoid storing raw content; store references and IDs.
- Choose a backend based on scale: SQLite for small apps, Redis/Postgres for production.

## Production Insight
Persistence is critical for long-running or multi-turn workflows where restarting from zero is too expensive or disruptive.
