from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


prompt1 = PromptTemplate(
    template = "Generate a comprehensive  Summary on the {topic} ",

    input_variables=["topic"]
    
)

prompt2 = PromptTemplate(
    template = "Generate 5 pints summary on the follwing text \n {text}"
    , input_variables=["text"]
)
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser();

chain= prompt1 | model | parser | prompt2 | model | parser 
result = chain.invoke({
    "topic":"Intersellar asteroids and metroids"
})

print(result)

chain.get_graph().print_ascii()

