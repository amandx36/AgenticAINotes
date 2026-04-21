# 🤖 AGENTS & TOOLS - Complete Expert Guide

## Table of Contents
1. [What Are Agents](#what-are-agents)
2. [Tool Design & Implementation](#tool-design--implementation)
3. [Tool Binding & Calling](#tool-binding--calling)
4. [Agent Architectures](#agent-architectures)
5. [Building an Agent from Scratch](#building-an-agent-from-scratch)
6. [Advanced Agent Patterns](#advanced-agent-patterns)
7. [Debugging Agents](#debugging-agents)
8. [Agent Interview Questions](#agent-interview-questions)

---

## What Are Agents

### Agent vs Chain

| Feature | Chain | Agent |
|---------|-------|-------|
| **Flow** | Predetermined | Dynamic (LLM decides) |
| **Flexibility** | Rigid, follows pipeline | Adaptive, thinks |
| **Tool Usage** | Fixed tool sequence | Chooses tools as needed |
| **Decision Logic** | None (hardcoded) | LLM reasoning |
| **Example** | Prompt → Model → Parser | "I need X tool, then Y tool" |

### Agent Components
```
AGENT ARCHITECTURE:

1. BRAIN (LLM)
   - Receives: task description, tool schemas, conversation history
   - Decides: which tool to call, with what parameters
   - Returns: thought process + tool choice

2. SENSORY SYSTEM (Tools)
   - Execute LLM's decision
   - Return results to LLM

3. MEMORY
   - Tracks decisions made
   - Learns from previous steps

4. EXECUTOR
   - Manages the loop
   - Handles retries and errors
```

### Agent Loop
```
START
  ↓
1. User provides goal: "Book a flight from NYC to LA"
  ↓
2. Agent (LLM) thinks: "I need to search flights, then book"
  ↓
3. Agent decides: Call search_flights_tool(from="NYC", to="LA")
  ↓
4. Tool executes: Returns list of flights
  ↓
5. Agent receives results: [Flight1, Flight2, Flight3]
  ↓
6. Agent thinks again: "Best option is Flight1, proceeding to book"
  ↓
7. Agent decides: Call book_flight_tool(flight_id=1, passengers=1)
  ↓
8. Tool executes: Returns booking confirmation
  ↓
9. Agent synthesizes: "Booked Flight1 for $450, confirmation #ABC123"
  ↓
10. Return final response to user
END
```

### Agent vs Simple Chain Example

```python
# CHAIN (Rigid):
chain = (
    get_weather_tool |
    suggest_restaurant_tool |
    book_table_tool |
    send_confirmation_tool
)
# Always runs all 4 steps, even if unnecessary

# AGENT (Intelligent):
agent = create_tool_calling_agent(
    llm, 
    tools=[get_weather, suggest_restaurant, book_table, send_confirmation]
)
# Agent decides: Is weather relevant? Do I need to book? What's the best path?
```

---

## Tool Design & Implementation

### Principle 1: Clear Purpose
Every tool has ONE clear job.

```python
# ❌ Bad: Too many responsibilities
@tool
def do_everything(task: str) -> str:
    """Does searching, calc, booking, and email"""
    if "search" in task:
        return search_web(task)
    elif "calc" in task:
        return calculate(task)
    # ...

# ✅ Good: Single responsibility
@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return duckduckgo_search(query)

@tool
def calculate_sum(numbers: list[float]) -> float:
    """Add a list of numbers"""
    return sum(numbers)
```

### Principle 2: Clear Schema (LLM's Perspective)

LLMs don't see the actual function. They see the **JSON schema**.

```python
from langchain.agents import tool

@tool
def calculate_total_price(
    item_count: int,
    price_per_item: float
) -> float:
    """Calculate total price for items.
    
    Args:
        item_count: Number of items to purchase
        price_per_item: Price per unit in dollars
    
    Returns:
        Total price as float (item_count * price_per_item)
    """
    return item_count * price_per_item

# LLM sees this JSON:
# {
#   "name": "calculate_total_price",
#   "description": "Calculate total price for items.\nTotal price as float (item_count * price_per_item)",
#   "parameters": {
#     "type": "object",
#     "properties": {
#       "item_count": {
#         "type": "integer",
#         "description": "Number of items to purchase"
#       },
#       "price_per_item": {
#         "type": "number",
#         "description": "Price per unit in dollars"
#       }
#     },
#     "required": ["item_count", "price_per_item"]
#   }
# }
```

### Principle 3: Good Error Handling

```python
@tool
def fetch_data(api_endpoint: str) -> str:
    """Fetch data from API endpoint"""
    try:
        response = requests.get(api_endpoint, timeout=5)
        response.raise_for_status()  # Raise for 4xx/5xx
        return response.json()
    except requests.Timeout:
        return "Error: API request timed out. Try again."
    except requests.HTTPError as e:
        return f"Error: HTTP {e.response.status_code}. {e.response.text}"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"
```

### Tool Types

#### 1. Query Tools (Search, Lookup)
```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search documentation for information"""
    results = db.full_text_search(query)
    return "\n".join([r["text"] for r in results[:5]])

@tool
def get_user_data(user_id: str) -> dict:
    """Get user profile information"""
    return database.fetch_user(user_id)
```

#### 2. Action Tools (Execute, Modify)
```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    email_service.send(to, subject, body)
    return f"Email sent to {to}"

@tool
def create_order(product_id: str, quantity: int) -> str:
    """Create a purchase order"""
    order = database.create_order(product_id, quantity)
    return f"Order created: {order.id} for {quantity}x {product_id}"
```

#### 3. Calculation Tools (Compute, Transform)
```python
@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price"""
    return price * (1 - discount_percent / 100)

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert between currencies"""
    rate = get_exchange_rate(from_currency, to_currency)
    return amount * rate
```

#### 4. Async Tools (Long-running)
```python
@tool
async def generate_report(report_type: str) -> str:
    """Generate analytics report (may take time)"""
    report = await async_report_generator.generate(report_type)
    return report

# Agent automatically handles async tools
```

---

## Tool Binding & Calling

### Step 1: Tool Binding (Register with LLM)

**What:** Tell the LLM what tools exist and how to use them.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import tool

# Define tools
@tool
def search(query: str) -> str:
    """Search the web"""
    return duckduckgo_search(query)

@tool
def calculator(expression: str) -> float:
    """Evaluate math expressions"""
    return eval(expression)

# Create LLM
llm = ChatOpenAI(model="gpt-4")

# Bind tools (register them)
tools = [search, calculator]
llm_with_tools = llm.bind_tools(tools)

# Now LLM knows: "I have access to search and calculator"
```

**Under the hood:**
```python
# bind_tools converts to:
llm_with_tools = ChatOpenAI(
    model="gpt-4",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        },
        # ... calculator tool similarly
    ]
)
```

### Step 2: Tool Calling (LLM Decides)

**What:** LLM receives a task and decides which tool to use.

```python
# User asks: "What's the weather in NYC?"

response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather in NYC?")
])

# LLM response:
# {
#   "tool_calls": [
#     {
#       "id": "call_abc123",
#       "name": "search",
#       "args": {"query": "weather NYC"}
#     }
#   ]
# }
```

**LLM thinks:** "I need to search for weather information"

### Step 3: Tool Execution (Actually Run)

```python
# Extract tool call from LLM response
tool_call = response.tool_calls[0]

# Map tool name to function
tools_map = {"search": search, "calculator": calculator}
tool_function = tools_map[tool_call["name"]]

# Execute with arguments
result = tool_function.invoke(tool_call["args"])
# result = "Weather in NYC: 72°F and sunny"
```

### Step 4: Result Injection (Loop Back)

```python
# Add result back to conversation
messages = [
    HumanMessage(content="What's the weather in NYC?"),
    AIMessage(content="", tool_calls=[...]),  # LLM's decision
    ToolMessage(content=result, tool_call_id=tool_call["id"])  # Tool result
]

# LLM sees the result and forms final response
final_response = llm_with_tools.invoke(messages)
# Output: "The weather in NYC is 72°F and sunny"
```

### Complete Tool Calling Flow
```
User: "What's the weather in NYC?"
  ↓
LLM (with bound tools): "I should use search_web tool"
  ↓
Tool Call: search_web(query="weather NYC")
  ↓
Tool Execution: search_web runs, returns weather data
  ↓
LLM receives result: "72F, sunny, low humidity"
  ↓
LLM generates response: "Weather in NYC today is 72°F and sunny"
  ↓
User: "Great! What should I wear?"
  ↓
LLM: "Based on the weather, wear light clothing"
```

---

## Agent Architectures

### Architecture 1: ReAct (Reasoning + Acting)

**How it works:**
```
Task: "Book cheapest flight from NYC to LA"

Step 1: THINK
"I need to search for available flights, compare prices by destination,
and book the cheapest one. Let me start by searching."

Step 2: ACT
Tool: search_flights("NYC", "LA")
Result: [Flight1 ($450), Flight2 ($520), Flight3 ($480)]

Step 3: THINK
"Flight1 is cheapest at $450. I should book this."

Step 4: ACT
Tool: book_flight("Flight1", passengers=1)
Result: Booking confirmation #ABC123

Step 5: OBSERVE
"Booking successful"

Step 6: OUTPUT
"Booked the cheapest flight ($450) with confirmation #ABC123"
```

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOpenAI(model="gpt-4")
tools = [search_flights, book_flight, check_prices]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Use tools to help the user."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_react_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Show reasoning steps
    max_iterations=10  # Prevent infinite loops
)

result = executor.invoke({"input": "Book cheapest flight NYC to LA"})
```

**Output with verbose=True:**
```
> Entering new AgentExecutor...
Thought: I need to search for flights from NYC to LA
Action: search_flights
Action Input: {"from": "NYC", "to": "LA"}
Observation: [Flight1: $450, Flight2: $520, Flight3: $480]
Thought: Flight1 is the cheapest. I should book it.
Action: book_flight
Action Input: {"flight_id": 1, "passengers": 1}
Observation: Booking confirmed #ABC123
Thought: Successfully booked the cheapest flight
Final Answer: Booked Flight1 for $450, confirmation #ABC123
> Finished AgentExecutor
```

### Architecture 2: Self-Ask with Search

**How it works:**
```
Question: "Who won the US presidential election in 2020?"

Agent thinks: "I need to know who won in 2020"
Questions itself: "Who was elected president in 2020?"
Searches for: "US president elected 2020"
Gets result: "Joe Biden"
Answers: "Joe Biden won the 2020 US presidential election"
```

```python
from langchain_core.prompts import PromptTemplate
from langchain_google_search import GoogleSearchAPIWrapper

llm = ChatOpenAI()
search = GoogleSearchAPIWrapper()

template = """Respond to the following questions as best you can. 
You have access to the following tools:

Google Search: useful for when you need to find recent information

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Google Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm | StrOutputParser()
```

### Architecture 3: Hierarchical Agents (Agent → Sub-agents)

```python
# Main agent delegates to specialized sub-agents

main_agent = create_agent(
    llm,
    tools=[
        book_flight_subagent,  # Handles flight booking
        book_hotel_subagent,   # Handles hotel booking
        car_rental_subagent    # Handles car rentals
    ]
)

# When user says: "Plan a trip to NYC"
# Main agent decides: "I need flight booking (→subagent1), 
#                      hotel booking (→subagent2),
#                      car rental (→subagent3)"
# Sub-agents handle their specific domains
```

---

## Building an Agent from Scratch

### Complete Example: Customer Support Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# ===== STEP 1: Define Tools =====

@tool
def search_kb(query: str) -> str:
    """Search knowledge base for FAQs and solutions"""
    # Simulate KB search (in reality, search RAG vector DB)
    kb = {
        "return": "30-day money-back guarantee. No questions asked.",
        "shipping": "Free shipping on orders over $50. Usual delivery: 3-5 days.",
        "refund": "Refund processed within 5-7 business days after return approved.",
        "damaged": "We ship with insurance. Report damage within 48 hours of delivery."
    }
    
    for key, value in kb.items():
        if key.lower() in query.lower():
            return value
    return "Sorry, I couldn't find information about that."

@tool
def check_order_status(order_id: str) -> dict:
    """Check status of a customer order"""
    # Simulate order lookup (in reality, query DB)
    orders = {
        "ORD-123": {"status": "Delivered", "date": "2024-03-08", "tracking": "UPS123"},
        "ORD-124": {"status": "In Transit", "date": "2024-03-10", "carrier": "FedEx"},
        "ORD-125": {"status": "Processing", "date": "2024-03-11"}
    }
    return orders.get(order_id, {"error": f"Order {order_id} not found"})

@tool
def initiate_refund(order_id: str, reason: str) -> str:
    """Initiate a refund for an order"""
    # Simulate refund initiation
    return f"Refund initiated for order {order_id}. Reason: {reason}. Refund ticket #REF-{order_id} created."

@tool
def escalate_to_human(issue: str, priority: str = "normal") -> str:
    """Escalate complex issue to human support"""
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"
    return f"Escalated to human support. Ticket: {ticket_id}. Priority: {priority}. You'll hear from us within 24 hours."

# ===== STEP 2: Create Agent =====

llm = ChatOpenAI(model="gpt-4")
tools = [search_kb, check_order_status, initiate_refund, escalate_to_human]

# Define prompt with system message
system_prompt = """You are a helpful customer support agent for an e-commerce company.
Your goal is to resolve customer issues efficiently.

Guidelines:
1. First try to solve using the knowledge base
2. For order issues, check the order status
3. For refunds, confirm the reason and process
4. For complex issues, escalate to human support
5. Always be empathetic and professional

Take action when needed, but ask for clarification if required."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Wrap with executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# ===== STEP 3: Use Agent =====

conversation_history = []

def chat(user_message: str):
    """Multi-turn conversation"""
    conversation_history.append(HumanMessage(content=user_message))
    
    result = agent_executor.invoke({
        "input": user_message,
        "chat_history": conversation_history,
        "agent_scratchpad": ""
    })
    
    return result["output"]

# Conversation
print(chat("Hi, what's your return policy?"))
# Agent: Uses search_kb → finds return policy
# Output: "30-day money-back guarantee..."

print(chat("I want to return order ORD-123 because it's damaged"))
# Agent: Uses search_kb (damage policy) + initiate_refund
# Output: "Refund initiated for ORD-123..."

print(chat("When will I get my money back?"))
# Agent: Uses search_kb (refund timeline)
# Output: "Refund processed within 5-7 business days..."
```

---

## Advanced Agent Patterns

### Pattern 1: Tool Retry with Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def unreliable_api(data: str) -> str:
    """Call unreliable external API with retries"""
    response = requests.get("https://api.example.com/data", params={"q": data})
    response.raise_for_status()
    return response.json()

# If API fails:
# Retry 1: Wait 2 seconds, try again
# Retry 2: Wait 4 seconds, try again
# Retry 3: Wait 8 seconds, try again
# If still fails, raise exception
```

### Pattern 2: Tool Chaining (Agent → Tool → Tool)

```python
# Some tools themselves use other tools internally

@tool
def book_flight_complete(from_city: str, to_city: str, date: str) -> str:
    """Complete flight booking process"""
    # This tool internally chains multiple steps
    
    # Step 1: Search flights
    flights = search_flights(from_city, to_city, date)
    
    # Step 2: Get best price
    best_flight = get_best_price(flights)
    
    # Step 3: Book it
    booking = book_flight(best_flight)
    
    # Step 4: Send confirmation
    send_email(f"Booking confirmed: {booking}")
    
    return f"Flight booked: {booking['confirmation']}"

# When agent calls this tool, all sub-steps happen automatically
```

### Pattern 3: Tool with Context Memory

```python
@tool
def recommend_product(category: str, context: dict = None) -> str:
    """Recommend product based on category and user history"""
    
    if context and "viewed_products" in context:
        # Use viewing history for personalization
        viewed = context["viewed_products"]
        # Recommend something different
    
    if context and "budget" in context:
        # Filter by budget
        budget = context["budget"]
    
    # Return recommendation
    return f"Based on your preferences, we recommend..."

# Usage with context
agent_executor.invoke({
    "input": "Recommend a laptop",
    "context": {
        "viewed_products": ["Dell XPS", "MacBook"],
        "budget": 1000
    }
})
```

### Pattern 4: Conditional Tool Execution

```python
from langchain_core.runnables import RunnableBranch

# Tool A: For complex issues
complex_issue_tool = create_tool_calling_agent(llm, complex_tools, prompt)

# Tool B: For simple issues
simple_issue_tool = create_tool_calling_agent(llm, simple_tools, prompt)

# Router: Decides which to use
issue_router = RunnableBranch(
    (lambda x: x["complexity"] == "high", complex_issue_tool),
    (lambda x: x["complexity"] == "low", simple_issue_tool),
    RunnablePassthrough()  # Default
)

# Usage
issue_router.invoke({
    "input": "My entire system is down",
    "complexity": "high"
})
# Uses complex_issue_tool with escalation capabilities
```

---

## Debugging Agents

### Problem 1: Agent Loops Forever

```python
# ❌ Problem
agent_executor = AgentExecutor(agent=agent, tools=tools)
# No max iterations, might loop infinitely

# ✅ Solution
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Stop after 5 iterations
    max_execution_time=60  # Stop after 60 seconds
)
```

### Problem 2: Agent Chooses Wrong Tool

```python
# ❌ Problem: Tool names too similar
@tool
def search_products():...
@tool
def search_knowledge_base():...
# Agent confused which to call

# ✅ Solution: Clear, distinct names
@tool
def search_product_catalog():...
@tool
def search_support_documentation():...
```

### Problem 3: Tool Execution Fails

```python
# ❌ Bad: Tool crashes, agent breaks
@tool
def risky_operation(data: str):
    result = dangerous_computation(data)  # May fail
    return result

# ✅ Good: Graceful error handling
@tool
def safe_operation(data: str):
    try:
        result = dangerous_computation(data)
        return result
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        return f"Operation failed: {str(e)}. Please try again."
```

### Debug Mode: See Agent Thinking

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # See all steps
)

result = agent_executor.invoke({"input": "Your query"})

# Output shows:
# > Entering new AgentExecutor...
# Thought: ...
# Action: ...
# Action Input: ...
# Observation: ...
# Final Answer: ...
```

### Tracing Agent Calls

```python
from langchain.callbacks import TracerCallback

tracer = TracerCallback()

result = agent_executor.invoke(
    {"input": "Your query"},
    callbacks=[tracer]
)

# Tracer logs every:
# - LLM call
# - Tool execution
# - Latency
# - Token usage
# - Errors
```

---

## Agent Interview Questions

**Q1: What's the difference between a Chain and an Agent?**
> **Chain:** Predetermined path, rigid, no thinking.
> **Agent:** Dynamic path, LLM decides, reasoning-based.
> Chains excel at straightforward tasks; agents handle complex, variable scenarios.

**Q2: Explain tool binding and tool calling.**
> **Binding:** Register tools with LLM as JSON schemas.
> **Calling:** LLM sees task, decides which tool to use, suggests parameters.
> **Execution:** Tool actually runs, result returned to LLM.
> **Loop:** LLM may call another tool or generate final answer.

**Q3: Design a customer support agent.**
> Tools needed:
> - `search_kb(query)`: Knowledge base search
> - `check_order(order_id)`: Order status
> - `process_refund(order_id, reason)`: Initiate refund
> - `escalate(issue, priority)`: Human handoff
> 
> LLM system prompt guides: "Try KB first, escalate if complex."

**Q4: How do you prevent agents from looping infinitely?**
> 1. Set `max_iterations` in AgentExecutor
> 2. Set `max_execution_time`
> 3. Well-defined tool termination conditions
> 4. Avoid tools that output tool calls themselves

**Q5: What makes a good tool?**
> - Single, clear responsibility
> - Descriptive name and docstring
> - Clear schema (type hints, descriptions)
> - Good error handling
> - Returns useful results (not bare data)

**Q6: Explain ReAct (Reasoning + Acting) pattern.**
> Agent: Think → Act → Observe → Reflect → Repeat
> 
> Example flow:
> Think: "Need to search flights"
> Act: Call search_flights()
> Observe: Get flight results
> Think: "Cheapest is Flight1"
> Act: Call book_flight()
> Done
> 
> Makes agent reasoning more transparent and controllable.

**Q7: Design a hierarchical agent system.**
> Main Agent (General queries)
> ├─ Flight Sub-agent (Flight-specific queries)
> ├─ Hotel Sub-agent (Hotel-specific queries)
> └─ Support Sub-agent (General questions)
> 
> When user input arrives, main agent routes to appropriate sub-agent.
> Each sub-agent optimized for its domain.

**Q8: How would you handle tool failures gracefully?**
> ```python
> try:
>     result = tool.invoke(args)
> except ToolError:
>     # Inform LLM of failure
>     next_tool = get_alternative_tool()
>     result = next_tool.invoke(args)
> except Exception:
>     # Tell agent: tool unavailable
>     return "Tool currently unavailable, trying alternative..."
> ```

**Q9: Compare tool-calling vs function-calling in agents.**
> **Tool-calling:** LangChain's abstraction, works with any LLM backend.
> **Function-calling:** OpenAI-specific, built into API.
> In practice, most LLMs support similar mechanisms; LangChain normalizes these.

**Q10: Design an agent with real-time adaptation.**
> Keep feedback loop:
> 1. Agent makes decision
> 2. Tool executes
> 3. Measure result quality
> 4. If poor, try alternative tool
> 5. If good, continue
> 6. Learn for future similar queries
> 
> Use vector DB to store: query → best_tool_path for future reference.

---

**Key Takeaway:** Agents are decision-making systems that use reasoning to choose actions. They're more powerful than chains but require careful tool design and error handling. Master agents and you can build sophisticated AI applications.
