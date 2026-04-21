from typing_extensions import TypedDict , Annotated , Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# select model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# schema
class Review(TypedDict):
    reviewer:Annotated[str,"Name of the reviewer"]
    rating: Annotated[int,"Rating (1-10)"]
    comment: Annotated[str,"Comment about the movie"]
    recommend : Annotated[Literal["yes","no"],"Whether the reviewer recommends the movie or not"]

Structured_output = model.with_structured_output(Review)

result = Structured_output.invoke(
    "Give me a structured review with reviewer, rating (1-10), and comment for the movie 3 Idiots"
)

print(result)
print(type(result))
print(result["reviewer"])
print(result["rating"])
print(result["comment"])

print(result["recommend"])