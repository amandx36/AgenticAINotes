# making the  Parallel  workFlow   without llm based 





    #           START
    #             |
    #             v
    #          Router
    #     /      |       \
    #    v       v        v
    # Node A  Node B   Node C
    #    \       |        /
    #     \      |       /
    #      ------+-------
    #            v
    #       Aggregator
    #            |
    #           END

from langgraph.graph import StateGraph  , START , END
from typing import  Dict , Any
from dotenv import load_dotenv
from typing import TypedDict