from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# define the prompt in one line 
prompt = PromptTemplate.from_template("{question}")


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # for enable  token level steaming 
    streaming = True 
)

parser = StrOutputParser()

# define the chain brother 

workFlow = prompt | llm | parser

# now run the model dude 
result = workFlow.invoke("which backend concept whould i focus so that i get the high package in spring boot ")
print(result)