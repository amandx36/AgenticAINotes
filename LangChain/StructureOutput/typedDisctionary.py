from typing import TypedDict

class Person(TypedDict):
    name : str 
    age : int 



# // make  object of type person 
new_person : Person={
    "name" : "Aman Deep",
    "age" : 30
}

print(new_person)