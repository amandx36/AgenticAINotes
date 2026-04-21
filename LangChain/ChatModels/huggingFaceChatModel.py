from langchain_ollama import ChatOllama 

llm = ChatOllama (

    model = "qwen3:4b",
    temperature = 0.2,
)

message = [
    ("system", "Your are a assistant that help  to guide me learn programming language, and you will answer my question in a simple way, and you will give me some example code to help me understand the concept better."),

    ("user","what is collection in java and how to use it ? ")
]
response = llm.invoke(message)
print(response)