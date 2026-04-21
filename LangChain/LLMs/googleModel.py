from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# select model
llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite")

result = llm.invoke(
    "What do you know about me? Also tell me which skills make the most money right now. with  freshers salary in india "
    "Tell me i  am currently learning web 2 devops and gen ai ml should i learn this all and i got approx 15 to 20 lpa yes or not jsut tell me dude "
)

print(result)
