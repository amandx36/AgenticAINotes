# Steps for Making MCP

## Common Workflow
This file captures the commands and steps for running an MCP server during development.

### Developer Commands
- `uv run fastmcp dev main.py` — start the MCP server in development mode
- `uv run fastmcp run main.py` — run the MCP server normally
- `uv run fastmcp install claude-desktop main.py` — install the running server into Claude Desktop

## Screenshot Explanation
![alt text](image-18.png)
This image shows the command flow or terminal session used to run the MCP server.

## Python to MCP Server Conversion
The following image shows how a Python application can be converted into an MCP server.

![alt text](image-19.png)

## Production Insight
For production, package the MCP server with clear startup scripts and document the commands for development, staging, and deployment.
