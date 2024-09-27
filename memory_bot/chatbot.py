from langchain_openai import AzureChatOpenAI   ## Using an LLM model from API calls
from langchain.prompts import ChatPromptTemplate, PromptTemplate ## A template to communicate with the bot
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import LLMChain

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



# from langchain import OpenAI, LLMChain
# from langchain.prompts import PromptTemplate

# Initialize the model (e.g., OpenAI's GPT-4)
# llm = OpenAI(model="gpt-35-turbo")
# AzureChatOpenAI LLm 
llm = AzureChatOpenAI(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_api_endpoint,
    deployment_name="gpt-35-turbo",  # Replace with your deployment name
    openai_api_version="2023-07-01-preview",  # For gpt-3.5-turbo
    model="gpt-35-turbo"
)


# Create a memory object that stores the conversation context
memory = ConversationBufferMemory()

# Initial detailed prompt template (first time the user asks a question)
initial_prompt_template = PromptTemplate(
    input_variables=["expression"],
    template="{expression}. Just tell me if this is true or not nothing else."
)

# Simplified template for subsequent inputs
followup_prompt_template = PromptTemplate(
    input_variables=["expression"],
    template="{expression}"
)

# Creating the chain to process the conversation
initial_chain = LLMChain(
    llm=llm, 
    prompt=initial_prompt_template, 
    memory=memory
)

followup_chain = LLMChain(
    llm=llm, 
    prompt=followup_prompt_template, 
    memory=memory
)

# Store whether we've already asked the initial detailed question
first_query = True

def ask_bot(expression):
    global first_query

    if first_query:
        # Send the detailed query first
        result = initial_chain.run(expression=expression)
        first_query = False  # After the first query, switch to follow-up mode
    else:
        # For subsequent queries, only send the expression without the full prompt
        result = followup_chain.run(expression=expression)

    return result

while True:
    # Test interaction
    exp_input = input("Enter your expression:- ")

    ask_bot(exp_input)

    # print(ask_bot("1 + 1 = 2"))  # First time
    # print(ask_bot("2 + 3 = 5"))  # Subsequent queries
