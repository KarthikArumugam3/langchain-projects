from langchain_openai import AzureChatOpenAI   ## Using an LLM model from API calls
from langchain.prompts import ChatPromptTemplate ## A template to communicate with the bot
# from langchain.output_parsers import StrOutputParser ## To parse the output of an LLM in a certain way (plain text, list, bullets, etc)
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve API key and endpoint
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Check if the environment variables are correctly set
if not azure_api_key or not azure_api_endpoint:
    raise ValueError("Azure OpenAI API key or endpoint is missing. Please check your .env file.")

## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

## Streamlit framework
st.title('Langchain Demo With Azure OpenAI API')
input_text = st.text_input("Search the topic you want")

# AzureChatOpenAI LLm 
llm = AzureChatOpenAI(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_api_endpoint,
    deployment_name="gpt-35-turbo",  # Replace with your deployment name
    openai_api_version="2023-07-01-preview",  # For gpt-3.5-turbo
    model="gpt-35-turbo"
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
# chain = prompt | llm 

if input_text:
    st.write(chain.invoke({'question': input_text}))
