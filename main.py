# flake8: noqa
from typing_extensions import TypedDict
from openai import OpenAI
from typing import Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = OpenAI()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # or ["POST"]
    allow_headers=["*"],
)

class ClassifyMessageResponse(BaseModel):
    isCodingQuestion: bool
    
class CodeAccuracyResponse(BaseModel):
    accuracyPercentage: str

class State(TypedDict):
    user_query: str
    llm_result: str | None
    accuracyPercentage: str | None
    isCodingQuestion: str | None
    
def classify_message(state: State):
    print("Classifying message")
    query = state['user_query']
    SYSTEM_PROMPT = """
    You are an helpfull AI Agent.
    Your job is detect if a user query is related to coding or not
    Return the response in specified JSON Boolean only.
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        response_format=ClassifyMessageResponse,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":query}
        ]
    )

    isCodingQuestion = response.choices[0].message.parsed.isCodingQuestion
    state["isCodingQuestion"] = isCodingQuestion
    return state
    
def route_query(state: State) -> Literal["general_query",'coding_query']:
    print("route_query")
    is_coding = state["isCodingQuestion"]
    
    if(is_coding):
        return "coding_query"
    
    return "general_query"

def general_query(state: State):
    print("General Query !")
    query = state["user_query"]
    
    SYSTEM_PROMPT = """
    You are a helpfull and smart AI Agent, 
    You have to answer the provided user query very percisly and smartly.
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":query}
            
        ]
    )
    answer = response.choices[0].message.content
    state["llm_result"] = answer
    return state
    
def coding_query(state: State):
    print("coding query")
    query = state["user_query"]
    
    SYSTEM_PROMPT = """
    You are a very precise and helpfull coding AI Assistant. 
    You are very skilled in solving the coding queries very smartly and respond the user with explainations and good quality answers.
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":query}
        ]
    )
    
    answer = response.choices[0].message.content
    state["llm_result"] = answer
    return state

def coding_validate_query(state: State):
    print("Validating message")
    query = state["user_query"]
    llm_result = state["llm_result"]
    SYSTEM_PROMPT = f"""
    You are an helpfull AI Agent.
    Your job is detect the accuracy of the coding question that is provided to you 
    
    User query : {query}
    Code : {llm_result}
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4.1-mini",
        response_format=CodeAccuracyResponse,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
        ]
    )

    accuracy = response.choices[0].message.parsed.accuracyPercentage
    state["accuracyPercentage"] = accuracy
    return state
    

graph_builder = StateGraph(State)

graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("route_query",route_query)
graph_builder.add_node("general_query",general_query)
graph_builder.add_node("coding_query",coding_query)
graph_builder.add_node("coding_validate_query",coding_validate_query)

graph_builder.add_edge(START,"classify_message")
graph_builder.add_conditional_edges("classify_message",route_query)

graph_builder.add_edge("general_query",END)
graph_builder.add_edge("coding_query", "coding_validate_query")
graph_builder.add_edge("coding_validate_query", END)

graph = graph_builder.compile()

class Agentic(BaseModel):
    prompt: str
@app.post("/")
def main(user: Agentic):
    _state = {
        "user_query": user.prompt,
        "accuracyPercentage": None,
        "isCodingQuestion": None,
        "llm_result": None
    }
    response = graph.invoke(_state)
    return {
    "response": response["llm_result"],
    "accuracy": response["accuracyPercentage"]}
