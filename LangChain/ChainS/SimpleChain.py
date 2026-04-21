from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template ="Generate  5 interesting facts about  {topic} "
    , input_variables=["topic"]
    
)
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()

chain = prompt | model | parser 

result = chain.invoke({"topic":"India"})
print(result)


# visualise the graph \
chain.get_graph().print_ascii()