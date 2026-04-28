from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    streaming=True
)

parser = StrOutputParser()

# Correct config
config = {
    "tags": ["llm apps", "report generation", "summarization"],
    "metadata": {
        "model": "gemini-3-flash-preview",
        "temperature": 0
    }
}

# FIX: map output of first chain into second prompt
chain = (
    prompt1 
    | llm 
    | parser 
    | (lambda text: {"text": text})  
    |prompt2
    | llm 
    | parser
)

# Run
result = chain.invoke(
    {"topic": "Artificial Intelligence"},
    config=config
)

print(result)