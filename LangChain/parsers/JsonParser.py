from cmd import PROMPT
from tempfile import template
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import JsonOutputParser, PromptTemplate 
from langchain_google_genai import GoogleGenerativeAI
# prompt template system
from langchain_core.prompts import PromptTemplate

load_dotenv()

# model
model = GoogleGenerativeAI(model="gemini-2.5-flash-lite")


parser = JsonOutputParser()


# Creaete prmpt template 

template = PromptTemplate