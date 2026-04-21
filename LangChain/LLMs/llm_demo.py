
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# select the model 
llm = OpenAI(model="gpt-4o-mini")


# this giving command to open ai 
result = llm.invoke("Tell me what you u know about me ")

print(result)