from dotenv import load_dotenv
from typing import TypedDict

# now for making the graph dude 
from langgraph.graph import StateGraph , END , START
# now for google llm models 
from langchain_google_genai import ChatGoogleGenerativeAI
# now making inmemeory checkPoints 
from langgraph.checkpoint.memory import InMemorySaver


# import for steaming 


# for converting into json 

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # for enable  token level steaming 
    streaming = True 
)

# now make the state 
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str



# making each function of node 

def generateJoke(state: JokeState):
    prompt = f"Generate a funny joke about {state['topic']}."
    full_response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content,end="",flush=True)
        full_response += chunk.content
    
    return {
        "joke":full_response
    }

# now making a function for generating the  explanation 

def generateExplanation(state: JokeState):
    prompt = f"Explain this joke: {state['joke']}"
    full_response = ""
    for chunk in llm.stream(prompt):
        print(chunk.content,end="",flush=True)
        full_response += chunk.content
    
    return {
        "explanation":full_response
    }

# make the graph 
 

graph = StateGraph(JokeState)

# now making the work Flow 
"""
        Start 
          |
        generate Joke 
          |
        generate Explanation 
          |
          END 


"""

graph.add_node('generate_joke',generateJoke)
graph.add_node('generate_Explanation',generateExplanation)


# now make the edges 
graph.add_edge(START,'generate_joke')
graph.add_edge('generate_joke','generate_Explanation')
graph.add_edge('generate_Explanation',END)



# now making the in memory check points 
checkPointer = InMemorySaver()


# enable graph level streaming 

workflow = graph.compile(checkpointer=checkPointer)

config1 = {"configurable":{"thread_id":"1"}}

for event in workflow.stream(
    {"topic":"pizza"},
    config = config1
):
    print(event)


