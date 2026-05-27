# MCP Introduction

## What Is MCP?
MCP is a protocol for connecting an AI host to a tool server or resource server using a common message format.
It is designed for interoperability between the AI brain and external capabilities.

## Architecture
MCP has three layers:
- **Data layer**: JSON-RPC 2.0 messages
- **Transport layer**: stdio, HTTP, WebSocket, SSE
- **Application layer**: tools, resources, prompts exposed by the server

## MCP Primitives
- **Tools**: actions the AI asks the server to perform
- **Resources**: dynamic structured data sources the AI can read
- **Prompts**: templated instructions that shape AI behavior

## Why JSON-RPC?
- Lightweight
- Bi-directional
- Transport agnostic
- Supports batching
- Supports notifications

## Transport Types
### Local Server
A local server runs on the same machine and typically communicates via stdio.
How it works:
1. Host launches the server subprocess.
2. Host writes JSON-RPC messages to stdin.
3. Server reads messages and writes responses to stdout.

### Remote Server
Remote servers commonly use HTTP and SSE for streaming.

## Production Insight
MCP is useful when tools should be isolated from the host or when multiple clients need access to a shared tool service.
