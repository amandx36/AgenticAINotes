#                     +---------+
#                     | START   |
#                     +---------+
#                          |
#         -----------------------------------
#         |                |                |
#    +---------+      +---------+      +---------+
#    |  COT    |      |  DOA    |      | Language|
#    +---------+      +---------+      +---------+
#    |  LLM    |      |  LLM    |      |  LLM    |
#    | (0-10)  |      | (0-10)  |      | (0-10)  |
#         |                |                |
#         -----------------------------------
#                          |
#                  +---------------+
#                  |  Final Eval   |
#                  +---------------+
#                  |      LLM      |
#                  +---------------+
#                          |
#                     +---------+
#                     |  END    |
#                     +---------+
from langgraph.graph import StateGraph, START, END
from typing import TypedDict , Annotated
import operator
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI


from pydantic import BaseModel , Field 


load_dotenv() 
# select model
llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite")





class EvaluationSchema(BaseModel):
    feedBack : str = Field(description= "Detailed feedBack of the essay dont add anything by your own and the feedback must be to the point and no hallucination")
    score : int = Field(description = " Score must be bettwen 0 to 10 ", ge=0 , le = 10 )

# now make the llm strictly follow  the schema 

structureModel = llm.with_structured_output(EvaluationSchema)

# now make a essay for the evaluation  provided by the user 
essay = """olar System Essay: About Our Solar System 

Everything orbits around the stronger force in space. It is not just the planets that orbit around the Sun, but the sun itself orbits along with all other planets. Our solar system is located inside the Milky Way galaxy. The galaxy has billions of other solar systems like ours. We are in one of the galaxy's four spiral arms. It takes around 230 million years for our solar system to orbit around the galactic center. The space outside our solar system is known as the Interstellar space. Two robotic spacecraft made their way out of our solar system. 

Out of all the solar systems we know, ours is the only one known to support life. Specifically, it is Earth which has life in it. The four giant planets have rings around it, out of which Saturn has the most spectacular rings. The eight planets of our solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune. Pluto, which was earlier classified as a planet, is now considered a dwarf planet. There are nearly 200 moons and countless asteroids varying in size. Let us understand about the planets of our solar system and their characteristics. 

Continue reading the Essay on our Solar System!
Mercury

Mercury is the smallest planet in our solar system and is the closest planet to the sun. It is only slightly larger than the Earth's moon. Due to its close proximity, its temperature is very high making life impossible on Mercury. It has the fastest orbital time with just 88 days to complete its revolution cycle. It travels in space at the speed of 29 miles per second! Its surface is rocky and hard much like our moon. Despite the proximity, it is not the hottest planet in the solar system. 
Venus 

Venus is also known as the evening star due to its bright shining look. Venus is called Earth's twin planet because of its similar size and density like Earth. Despite this, the two planets have radical differences. It is observed to be bright due to its carbon dioxide filled thick atmosphere. It also has the youngest surface than any planet anywhere like 150 million years. Scientists don't know what caused Venus to restructure itself. Venus rotates backward on its axis unlike most planets in our solar system. This means that on Venus, the Sun rises in the west and sets in the east, opposite of Earth. 
Earth

Earth is also known as the blue planet due to its blue appearance from outer space. It is due to the 70% water covering earth's surface. Earth is the only planet that supports life and possibly the most diverse and complex. It is the biggest of the four planets close to the sun. A day on Earth is 24 hours and it takes around 365 days to complete one revolution. It has 78% oxygen making it the perfect planet to breathe and live inz. There are so many satellites orbiting around the Earth to observe and instill various purposes. It is our home sweet home!
Mars

Mars is the fourth planet from the sun and it is also called the Red Planet due its red appearance. It is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes and with evidence that it was even more active in the past. Mars is one of the most explored planets in our solar system. It was discovered that Mars was much wetter and warmer with a thicker atmosphere billions of years ago. It is observed that no other planet captured human imagination other than Mars due its supporting evidence of habitat. 
Jupiter

Also known as the Vacuum Cleaner of our Solar System, Jupiter is one large gas giant! It is the largest planet in our solar system with twice the mass of all the other planets combined. It has an iconic red spot on its surface which is larger than Earth. Scientists found that it is a raging storm in Jupiter for hundreds of years. Jupiter completes a rotation every 10 hours but takes 12 years to complete one revolution. It doesn't have a solid surface and is filled with gaseous substances only. Jupiter has more than 75 moons!
Saturn 

Saturn too is a gas giant and is the second largest planet in our solar system. It has spectacular rings which are made of chunks of ice and rocks. Saturn has about 82 moons some of which are yet to be discovered. It has an unlikely condition to support life as it is made of helium and hydrogen mostly with no surface. However scientists believe that some of Saturn's moon can support life. It takes around 29 earth years for Saturn to complete one orbit around the sun. 
Uranus

Uranus is the 7th planet from the Sun and has the third largest diameter from our solar system. Uranus takes about 17 hours to finish a rotation and 84 earth years to complete one orbit around the sun. It is made of ammonia, methane and icy materials. It is also called the ice giant. Uranus too rotates like Venus, but slightly with a difference. It rotates on its side unlike other planets. It was discovered in 1789 by Herschel, who initially thought it was a comet or a star. 
Neptune

The eighth planet from the sun is Neptune, a dark, cold and distant planet. Neptune takes around 165 years to orbit the Sun. It is the only planet out of the eight planets which cannot be seen through naked eyes. Neptune too cannot support life as it is a gas giant with no possible atmosphere that could support life. 
Pluto

The ninth planet from the sun which is considered as a dwarf planet, it is the smallest planet than Mercury. It is smaller than our moon and has a heart shaped ice glacier. It has blue skies and rocky mountains and even snows – but in red color. Pluto is away from us and located in the Kuiper belt. Pluto's surface is far too cold making life impossible on it. 
Other Elements in Our Solar System

    Satellites: These are man-made particles that hover in space with a task of observing and exploring various extraterrestrial bodies. 
    Comets: Comets are frozen leftovers from the formation of solar systems. It is composed of dust, rock and ice. 
    Meteoroids: Meteoroids are small size asteroids ranging in size from a grain to meters. 

    Asteroids: Asteroids are rocky or icy bodies whose size ranges from meters to the size of a dwarf planet. It is a minor planet of the inner solar system with no atmosphere. 

    Space Dust: Anything smaller than the size of a meteoroid is classified as space dust. 

Solar System Essay Conclusion 

Our solar system is a complex and mysterious place. Though many spacecraft hovered and are still hovering in outer space, it is practically impossible for humans to wander the space. The distance is quite huge and is measured by Astronomical units (AU). 1 AU is the distance between the Earth and the Sun. Our solar system is roughly 180 Astronomical units in diameter. Other than many planets and moons, the solar system contains meteoroids, asteroids and comets which wander around in the empty space. Space study is an ideal way to understand more about the solar system and other fantastic mysteries of our universe. In the endless cosmos, all we are is just a speck! 
10 Pointers to Write Essay on Solar System 

    Our solar system is one among the billions of other solar systems that exist in the milky way galaxy.
    It takes around 230 million years for our solar system to orbit around the galactic center.
    The Sun is the biggest star of our solar system and it is the center with the highest gravity, around which other planets revolve. 
    Other elements in our solar system include planets, moons, comets, asteroids and other space particles.
    Our solar system consists of 8 planets and one dwarf planet.
    Out of these 8 planets, four are terrestrial planets and the remaining are gas giants.
    The eight planets of our solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune.
    Each and every planet has its unique characteristics and revolutionary pattern.
    Space study is an ideal way to understand more about the solar system and other fantastic mysteries of our universe.
    In the endless cosmos, all we are is just a speck!
"""



# now make a prompt for this 

prompt = f"Evaluate the following essay  Be specific and constructive in your feedback. Essay: {essay}"


# invoke the  llmt 
structured_output = structureModel.invoke(prompt)

#dude now define the state dude 
# clarity of tought (coa)
# depth of analysis (doa)

load_dotenv()  # Load environment variables from .env file


# defining the state 
class ExamState(TypedDict):
    essay : str 
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float


# now llm predict the score 




# now define the functions for each node
def evaluate_language(state : ExamState) -> StateGraph:
    # logic to calculate language proficiency
     prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
     output = structureModel.invoke(prompt)
     return {"language_feedback": output.feedBack, "language_score": [output.score]}

# for depth of analysis
def evaluate_analysis (state : ExamState) -> StateGraph:
     
     prompt = f'Evaluate the depth  of  analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
     output = structureModel.invoke(prompt)
     return {"analysis_feedback": output.feedBack, "analysis_score": [output.score]}


# for language Score 

def evaluate_thought(state : ExamState) -> StateGraph:
    # logic to calculate language proficiency
    
    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}'
    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}


# now  make the final evaluation function 
def Final_Evaluation(state : ExamState) -> StateGraph:
    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = llm.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}



# now dude make the graph 

graph = StateGraph(ExamState)

# Create node 
graph.add_node('Calculate_coa', evaluate_language)
graph.add_node('Calculate_doa', evaluate_analysis)
graph.add_node('Calculate_language_proficiency', evaluate_thought)
graph.add_node('Final_Evaluation', Final_Evaluation)


# now dude make  the edges 

graph.add_edge(START , 'Calculate_coa')
graph.add_edge(START,'Calculate_doa')
graph.add_edge(START,'Calculate_language_proficiency')

# now all three is converted to summary brother 

graph.add_edge('Calculate_coa', 'Final_Evaluation')
graph.add_edge('Calculate_doa', 'Final_Evaluation')
graph.add_edge('Calculate_language_proficiency', 'Final_Evaluation')


 # now the final evaluation is connecte to end dude 

graph.add_edge('Final_Evaluation', END)

workFlow = graph.compile()


 # make the intial workFlow state 



output_state = workFlow.invoke()


print("Final State is this \\n", output_state)

print("\n\n\n\n")
print("Final Score : ", output_state['final_score'])