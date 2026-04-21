from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# model
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")

# template 1
template1 = PromptTemplate(
    template="Write a brief on {topic} in depth", input_variables=["topic"]
)

# template 2
template2 = PromptTemplate(
    template="Make this into {numberOfPoints} important points:\n{text}",
    input_variables=["text", "numberOfPoints"],
)

# parser
parser = StrOutputParser()

# chain (FIX: convert string → dict before template2)
chain = (
    template1
    | model
    | parser
    | (lambda x: {"text": x, "numberOfPoints": 5})
    | template2
    | model
    | parser
)

# invoke
result = chain.invoke({"topic": "black hole"})

print(result)
