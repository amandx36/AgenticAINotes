        #         +--------+
        #         | START  |
        #         +--------+
        #          /   |   \
        #         /    |    \
        #        /     |     \
        #   +-----+  +-----+  +-----+
        #   | SR  |  | b/s |  | bpb |
        #   +-----+  +-----+  +-----+
        #       \        |        /
        #        \       |       /
        #         \      |      /
        #          \     |     /
        #          +-------------+
        #          |  SUMMARY    |
        #          +-------------+
        #                 |
        #                 v
        #            +--------+
        #            |  END   |
        #            +--------+


# for working in parllel chain so we have to send the and recive data which is changed in the state to the next node
from langgraph.graph import StateGraph  , START , END
from typing import  Dict , Any
from dotenv import load_dotenv
from typing import TypedDict

# make  teh state first 

class  CricketState(TypedDict):
    run : int 
    balls : int 
    fours : int 
    sixes : int 

    StrikeRate  : float 
    bps : float 
    boundary_percentage : float 


# now make all the functions dude of the logic brother 

def calculate_SR(state : CricketState) -> CricketState:
   state['StrikeRate'] = ( state['run'] / state['balls'] ) * 100
   return {'StrikeRate':state['StrikeRate']}

def calculate_bps(state : CricketState) -> CricketState:
    state['bps'] =  state['balls']/(state['fours'] + state['sixes'])
    return {'bps':state['bps']}

def calculate_boundary_percentage(state : CricketState) -> CricketState:
    state['boundary_percentage'] = ((state['fours'] + state['sixes']) / state['balls']) * 100
    return {'boundary_percentage':state['boundary_percentage']}

def summary(state : CricketState) -> CricketState:
    print(f"Runs: {state['run']}")
    print(f"Balls: {state['balls']}")
    print(f"Fours: {state['fours']}")
    print(f"Sixes: {state['sixes']}")
    print(f"Strike Rate: {state['StrikeRate']:.2f}")
    print(f"BPS: {state['bps']:.2f}")
    print(f"Boundary Percentage: {state['boundary_percentage']:.2f}%")
    return state

# make  the graph 


graph = StateGraph(CricketState)


# make 4 node dude that is SR , b/s  bpb and summary

graph.add_node('calculate_SR',calculate_SR)
graph.add_node('calculate_bps',calculate_bps)
graph.add_node('calculate_boundary_percentage',calculate_boundary_percentage)
graph.add_node('summary',summary)



# now i have to define the edges dude 
graph.add_edge(START, 'calculate_SR')
graph.add_edge(START, 'calculate_bps')
graph.add_edge(START, 'calculate_boundary_percentage')

# now connect all the 3 nodes to summary

graph.add_edge('calculate_SR', 'summary')
graph.add_edge('calculate_bps', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')

# now connect summary to END 

graph.add_edge('summary', END)


# now compile the graph 

workFlow = graph.compile()


print(workFlow)


# now execut our project 

intial_state = CricketState(run=120, balls=100, fours=10, sixes=5, StrikeRate=0.0, bps=0.0, boundary_percentage=0.0)
outPut_State = workFlow.invoke(intial_state)
print(outPut_State)