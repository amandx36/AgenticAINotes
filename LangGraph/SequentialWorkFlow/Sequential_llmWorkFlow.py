from langgraph.graph import StateGraph  , START , END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import  Dict , Any 
# key value of any pair [string , Any ]
from dotenv import load_dotenv 
from typing import TypedDict 


load_dotenv()

model = ChatGoogleGenerativeAI(
 model="gemini-2.5-flash-lite"   
)


# create state 
class LLmState(TypedDict):
    question: str 
    answer : str 

def llm_question(state :LLmState)-> LLmState:
    # extract the question 
    question = state['question']
    #form a prompt 
    prompt = f" you are a helpfull assistance now answer the following in easy way {question}"

    # send it to the llm 
    answer = model.invoke(prompt).content

    print("for debugging",answer)

    # update the prompt 
    state['answer'] = answer

    # now return the state 
    return state 
    

# create a graph dude 
graph = StateGraph(LLmState)

# now add the node dude 
graph.add_node('llm_ques',llm_question)

# add the edges 
graph.add_edge(START , "llm_ques")
graph.add_edge("llm_ques",END)
# compile it 
workFlow =  graph.compile()

# execute 
initialState= {'question':'How far is moon from the earth'}

finalState = workFlow.invoke(initialState)

print(finalState)
