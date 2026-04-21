from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)

# ---------------- STATE ----------------
class LLmState(TypedDict):
    topic: str
    outline: str
    Blog: str
    score: str   # keep string (safe)

# ---------------- NODES ----------------
def gen_outline(state: LLmState) -> LLmState:
    topic = state['topic']

    prompt = f"Generate a clear blog outline for: {topic}"
    outline = model.invoke(prompt).content

    print("DEBUG outline:", outline)

    return {
        **state,
        "outline": outline
    }

def gen_blog(state: LLmState) -> LLmState:
    topic = state['topic']
    outline = state['outline']

    prompt = f"Write a blog on: {topic} using this outline:\n{outline}"
    blog = model.invoke(prompt).content

    print("DEBUG blog:", blog)

    return {
        **state,
        "Blog": blog
    }

def score_blog(state: LLmState) -> LLmState:
    blog = state['Blog']

    prompt = f"Give a score (1-10) for this blog:\n{blog}"
    score = model.invoke(prompt).content

    return {
        **state,
        "score": score
    }

#  GRAPH ----------------
graph = StateGraph(LLmState)

graph.add_node("gen_outline", gen_outline)
graph.add_node("gen_blog", gen_blog)
graph.add_node("score_blog", score_blog)

graph.add_edge(START, "gen_outline")
graph.add_edge("gen_outline", "gen_blog")
graph.add_edge("gen_blog", "score_blog")
graph.add_edge("score_blog", END)

workflow = graph.compile()

# ---------------- RUN ----------------
initialState = {
    "topic": "How to achieve 20-30 LPA as a fresher in software engineering"
}

finalState = workflow.invoke(initialState)

print("\nFINAL STATE:", finalState)
print("Final Score:", finalState["score"])