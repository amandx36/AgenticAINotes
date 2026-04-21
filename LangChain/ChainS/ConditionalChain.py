# ================================
# 📦 Imports
# ================================
from dotenv import load_dotenv

# LLM
from langchain_google_genai import GoogleGenerativeAI

# Prompt + parsing
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# Runnables (core building blocks of chains)
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

# Pydantic for structured output
from pydantic import BaseModel, Field
from typing import Literal



# 🔐 Load environment variables

load_dotenv()


# ================================
# 🧠 Define structured output schema
# ================================
class FeedBack(BaseModel):
    # Model will classify sentiment into one of these
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="Sentiment of the feedback"
    )


# ================================
# 🔄 Parser to convert LLM output → Pydantic object
# ================================
parser2 = PydanticOutputParser(pydantic_object=FeedBack)


# ================================
# 🤖 Initialize LLM
# ================================
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ================================
# 🧾 Prompt 1 → Sentiment Classification
# ================================
prompt1 = PromptTemplate(
    template="""
Classify the following feedback into one of:
positive, negative, neutral

{format_instructions}

Feedback:
{feedback}
""",
    input_variables=["feedback"],
    
    # Inject parser instructions so model outputs JSON-like structure
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)


# ================================
# 🔗 Chain 1 → Classification
# Output → FeedBack object
# ================================
classifier_chain = prompt1 | model | parser2


# ================================
# 🧾 Prompt 2 → Positive response
# ================================
prompt_positive = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)


# ================================
# 🧾 Prompt 3 → Negative response
# ================================
prompt_negative = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)


# ================================
# 🔁 Step: Preserve BOTH
# - original feedback
# - classification result
# ================================
parallel_step = RunnableParallel({
    # Run classification
    "sentiment": classifier_chain,

    # Pass original input forward unchanged
    "feedback": RunnableLambda(lambda x: x["feedback"])
})


# ================================
# 🌿 Branching Logic
# ================================
branch = RunnableBranch(

    # ✅ Positive branch
    (
        lambda x: x["sentiment"].sentiment == "positive",
        prompt_positive | model | StrOutputParser()
    ),

    # ❌ Negative branch
    (
        lambda x: x["sentiment"].sentiment == "negative",
        prompt_negative | model | StrOutputParser()
    ),

    # ⚪ Default → Neutral case
    RunnableLambda(
        lambda x: "Neutral feedback received, no response needed."
    )
)


# ================================
# 🔗 Final Chain
# ================================
chain = parallel_step | branch


# ================================
# 🚀 Run the chain
# ================================
result = chain.invoke({
    "feedback": "I am very happy with the product and the service provided by the company"
})


# ================================
# 📤 Output
# ================================
print("\n===== FINAL RESPONSE =====\n")
print(result)


# ================================
# 📊 Visualize execution graph
# (requires: pip install grandalf)
# ================================
chain.get_graph().print_ascii()