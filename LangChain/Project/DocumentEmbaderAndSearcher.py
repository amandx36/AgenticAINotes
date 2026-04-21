from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Create embeddings model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

# Your text
text =[
     "What is collection in java and how to use it?",
     "Where  you find the lang chain documentation ?",
     "What is the best way to learn programming language ?",
     "What is the best way to learn programming language ?",]

# Create vector store
vectorStore = InMemoryVectorStore.from_texts(
    text ,
    embedding=embeddings
)


# now search the input text vector from the closest one dude 
input_text = "How i become a billionaire ?"


# Retriving the closest text from the vector store 


retriever = vectorStore.as_retriever()
retriver_para = retriever.invoke(input_text)

print("The closes text to input text using embeded search its ID is this  :) ",retriver_para[0].id)
print("And the text is this  :) ",retriver_para[0].page_content)

