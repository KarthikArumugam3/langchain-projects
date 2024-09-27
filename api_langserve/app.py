# create apis

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
# from langchain_openai import ChatOpenAI
from langserve import add_routes  # Routes to communicate with diff llms
import uvicorn 
import os 
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Create Fastapi
app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)

# OpenAI API
model = AzureChatOpenAI(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_api_endpoint,
    deployment_name="gpt-35-turbo",  # Replace with your deployment name
    openai_api_version="2023-07-01-preview",  # For gpt-3.5-turbo
    model="gpt-35-turbo"
)

add_routes(
    app,
    model,
    path="/openai"
)



## Opensource Ollama model
llm = Ollama(model="llama2")

### Prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/OpenAI/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/Ollama/poem"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)

