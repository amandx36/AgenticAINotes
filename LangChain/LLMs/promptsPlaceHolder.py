from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
])

chatHistory = []

# load chat history from file  or a database 

with open("chatHistory.txt", 'r') as f:
    chatHistory.append(f.readline())

print(chatHistory)


# create response 
prompt  =  chat_template.invoke({"chat_history": chatHistory, "input": "where is my gf ?"})

print (prompt)