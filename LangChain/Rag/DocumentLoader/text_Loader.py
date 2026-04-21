from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
# importing text loader 
from langchain_community.document_loaders import TextLoader
#importing parser 
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 



parser = StrOutputParser()


load_dotenv()


#loading the text file using text_loader 

loader  = TextLoader("cricket.txt",encoding="utf-8")

# making object of text loader 
documentss = loader.load();

print(documentss)

# model part dude 



model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template = "Write a summary for this {poem}",
    input_variables = ["poem"] 
)


#  make the chian 
chain = prompt | model | parser 
result = chain.invoke({"poem": documentss[0].page_content})

print(result)


