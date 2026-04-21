from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")

doc = loader.load()
print(doc)

print(type(doc))
print("Next")

print(type(doc[0]))
print("page content ")
print(doc[0].page_content)

print("metadata"   )
print(doc[0].metadata)