# LangGraph Module 01

## Overview
This note discusses agentic AI, LangChain, LangGraph, and core architecture patterns for autonomous workflows.

## Generative AI vs Agentic AI
- **Generative AI**: creates content (text, image, code, audio) based on learned distributions.
- **Agentic AI**: a behavior that takes goals and executes steps to complete them autonomously.

## Agentic AI Characteristics
1. Autonomy
2. Goal orientation
3. Planning
4. Reasoning
5. Adaptability
6. Context awareness

## Autonomy
Autonomy means the system can make decisions and act toward a goal without step-by-step instructions.

### Control Mechanisms
- Permission scope
- Human-in-the-loop
- Override controls
- Guardrails and policies

## Goal-Oriented AI
Goal-oriented AI stores and updates goals, tracks progress, and chooses actions to complete objectives.

## Planning
Planning breaks a high-level goal into subgoals and selects the best execution path.

## Agent Architecture
Components:
- Brain
- Orchestrator
- Tools
- Memory
- Supervisor

## Workflow vs Agentic Application
| Feature | Workflow | Agentic App |
|---|---|---|
| Control | Fully manual | Shared with AI |
| Flexibility | Low | High |
| Decision making | No | Yes |
| Adaptability | No | Yes |
| Complexity | Simple | Advanced |

## LangChain
LangChain connects LLMs with tools, memory, and data using chains and agents.

## Prompt Chaining Types
1. Sequential flow
2. Routing / dynamic flow
3. Parallel execution
4. Orchestrator/workers
5. Evaluator/optimizer

## Interview Angle
Explain LangChain as a framework for composing LLMs with external data, tools, and workflows, and LangGraph as the graph-based orchestration layer.
