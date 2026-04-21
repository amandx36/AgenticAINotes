

Generative Ai  .:)  refresrs  class of ai model where that can create new contents  such as text image audio code  or video 

generative is about learning the distribution of data so that it can generate a new sample from it 

or summarise it is 
    generative ai is capability and agentic ai is a beheviour  

 apps like chatgpt google geminie claude and grok 

 nano banaa 


 Traditional AI  :) finding patterns in data and giving 
                        like spam and ham classifier 
                        


application of generative area 
            1 create and business writing 
            2 Softwae developlment 
            3 customer support 
            4 Education 
            5 Designing 


Agentic Ai type of ai that take up a task or goal from a usr and than  work toward compleitng it on its own with minimal human guidance 


it plasn take action adapts to chnges and seek help only when necessary 

characteristic of agentic ai 

1 autonomous 
2 Goal oriented 
3 planning 
4 reasoning 
5 adaptability 
6 context awarness

1) Autonomy

Autonomy refers to the AI system’s ability to make decisions and take actions on its own to achieve a given goal, without needing step-by-step human instructions.

Our AI recruiter is autonomous
It’s proactive
Autonomy in multiple facets
a. Execution
b. Decision making
c. Tool usage

Autonomy can be controlled

a. Permission Scope – Limit what tools or actions the agent can perform independently. (Can screen candidates, but needs approval before rejecting anyone.)

b. Human-in-the-Loop (HITL) – Insert checkpoints where human approval is required before continuing. (Can post this JD?)

c. Override Controls – Allow users to stop, pause, or change the agent’s behaviour at any time. (pause screening process due to last resume processing.)

d. Guardrails / Policies – Define hard rules or ethical boundaries the agent must follow. (Never schedule interviews on weekends)

Autonomy can be dangerous

a. The application autonomously sends out job offers with incorrect salaries or terms.

b. The application shortlists candidates by age or nationality, violating anti-discrimination laws.

c. The applications spending extra on LinkedIn ads. 



2) Goal-Oriented (Simple Explanation)



A goal-oriented AI focuses on a specific objective and keeps working toward it continuously.
It doesn’t just respond to one prompt → it acts step-by-step until the goal is achieved.
Key Points
Goals guide autonomy (AI knows what to achieve)
Goals can have constraints (e.g., skills, experience)
Goals are stored in memory (so AI remembers progress)
Goals can be updated/changed
Example from image

AI goal: “Hire a backend engineer”

Constraints: experience, tech stack
Tracks progress: job posted, applications received, interviews scheduled
Planning

3) Planning is the agent’s ability to break down a high-level goal into a structured sequence of actions or subgoals and decide the best path to achieve the desired outcome.

Step 1: Generating multiple candidate plans

Plan A: Post JD on LinkedIn, GitHub Jobs, AngelList
Plan B: Use internal referrals and hiring agencies

Step 2: Evaluate each plan

Efficiency (Which is faster?)

Tool Availability (Which tools are available)
Cost (Does it require premium tools?)
Risk (Will it fail if we get no applicants?)
Alignment with constraints (remote-only? budget?)

Step 3: Select the best plan with the help of:

Human-in-the-loop input (e.g., “Which of these options do you prefer?”)
Organizational policy (e.g., “Favor low-cost channels first”)


components of ai agent 

A Brain 
B orchestrate 
C Tools 
D Memory 
E Supervisor 


Differnet between workflow and agentic application 

1. Workflow (Deterministic System)

Fixed sequence of steps (predefined logic)
No real decision-making
Same input → same output path
You control everything

👉 Every step is hardcoded


2. Agentic Application (Autonomous System)

Has goal + autonomy + planning
Can decide what to do next dynamically
Uses tools, memory, reasoning
Can adapt if situation changes


QUICK TABLE 


| Feature         | Workflow     | Agentic App    |
| --------------- | ------------ | -------------- |
| Control         | Fully manual | Shared with AI |
| Flexibility     | Low          | High           |
| Decision Making | No           | Yes            |
| Adaptability    | No           | Yes            |
| Complexity      | Simple       | Advanced       |


glue code :) more glue code worst the application is 


State = a shared, structured key–value object that flows through all nodes in a LangGraph and gets updated at each step.


example :) state = {
    "query": "Hire backend engineer",
    "plan": [],
    "candidates": [],
    "selected": None
}

Planner node → updates "plan"
Search node → fills "candidates"
Decision node → sets "selected"


we can  make a subgrapgh dude 
 

LANGCHAIN


 LangChain is a framework for building applications powered by Large Language Models (LLMs) by connecting them with data, tools, and workflows.



 WorkFlow :) series of task which can operated to acive the goal 


 llm WorkFlow :) during the executation of step , if any step use the llm work flow is know as llm work flow 
    
        work flow are a step by step using which we can build complex application llm application 

        each step in a workFlow perform a distinct task - such a prompting reasoning tool calling memory acces or decision making 

    workFlow can be linear parllel branched or looped 


Types of prompt channing 

1. Prompt Chaining (Sequential Flow )    :)  


[Input]
   ↓
[LLM Call 1]
   ↓
[Output 1]
   ↓
[Gate / Condition]
   ├── Pass → [LLM Call 2] → [Output 2] → [LLM Call 3] → [Final Output]
   └── Fail → [Exit]

2. Roting (Dynamic Flow )


[Input]
   ↓
[LLM Router]
   ├── Route 1 → [LLM Call 1] → [Output]
   ├── Route 2 → [LLM Call 2] → [Output]
   └── Route 3 → [LLM Call 3] → [Output]


One input → multiple possible paths
Router decides best path
Only one path executes

3. parallelization 


        |----------> llm call 1 ---------
        |                               |
        |                               |
        |                               |
input -------------->llm call 2 -------------> Agrregator -----> output 
        |                               |
        |                               |
        |------------> llm call 3 -------    
    
    dividing the input into subpart and than dividde it into others work 



4. Orchestrator Workers 


            based on input what to perform task  and where to perform 



                    |----------------->> llm call 1 ---
                    |                                   |
                    |
input ----->  Orchestrator-------------  llm call 2 --->>  Synthesizer -----> outpiut 
                    |
                    |                                      |
                    |                                      |
                    ---------------------llm call 3  -----|



5 Evaluator Optimizer 


                            |-------------------|
                                                |
                            |                   |
                            |                   |
                            |            llm call Evaluator [evaluate is this right y]      ------------------------------>[ output ]
input ------> llm call generator                [yes than go forward / other wise go to back  and reevaluate it ]
                            |                   |
                            |                   |
                            |                   |
                            |--------------------


Graphs , Nodes and Edges 


Nodes a :)  python function where  we write the logic dude 
Edges :) when yu have to execute your node 




States :)) 


State in LangChain = the data/context that is passed between steps (chains, agents, tools) during execution in key value pair .


any moment node has the acees of executation 

it is mutable 

Easy Understanding
It is what your app knows at a given moment
It keeps track of:
Input
Output
Intermediate results
History

:))  Think: “current working data”

state = {
    "input": "Summarize this document",
    "documents": [...],
    "summary": None,
    "final_answer": None
}

Reducer :)) 

Reducer = a function that controls how the state is updated when multiple nodes try to change the value .

Why Reducer is Needed

In LangGraph:

Multiple nodes can update the same state key
Especially in:
Parallel execution
Loops

👉 So conflict happens: which value should win?

➡️ Reducer solves this



LangGraph Execution Model is inspired by google Pregel 

google pregel is 

A system by Google for large-scale graph processing
Works in super-steps (round-based execution)


1. Graph Definition

You define:

State Schema → what data flows
Nodes → functions/tasks
Edges → connections between nodes

👉 Basically: structure of your system

2. Compilation
Call .compile() on the graph
Validates:
Nodes
Edges
State

👉 Prepares graph for execution

3. Invocation

Run using .invoke(initial_state)
Initial state is sent to starting node

👉 This starts the workflow

4. Super-Steps (Execution Rounds)

👉 Execution happens in steps/rounds, not randomly

Nodes execute
Update state
Pass results to next nodes


5. Message passing and Node Activation 
  The messages are passed to downStream nodes via edges 
  Nodes that receive messages beocme active fo the next round 
  

  