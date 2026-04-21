from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = GoogleGenerativeAI(model="gemini-2.5-flash-lite")
model2 = GoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt1 = PromptTemplate(
    template="Generate comprehensive detailed notes on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 quizzes with options and answers on {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the following into comprehensive notes:\n{notes}\n{Quiz}",
    input_variables=["notes", "Quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "Quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic": "Interstellar asteroids and meteoroids"})

print(result)

chain.get_graph().print_ascii()