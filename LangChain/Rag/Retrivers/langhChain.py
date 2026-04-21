from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

from langchain_community.retrivers import WikipediaRetriever

#make object of it udde 

retriver = WikipediaRetriever()