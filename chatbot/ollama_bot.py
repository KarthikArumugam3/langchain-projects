from langchain_openai import AzureChatOpenAI   ## Using an LLM model from API calls
from langchain.prompts import ChatPromptTemplate ## A template to communicate with the bot
# from langchain.output_parsers import StrOutputParser ## To parse the output of an LLM in a certain way (plain text, list, bullets, etc)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama ## For thirdparty integration or opensource llms
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()



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
st.title('Langchain Demo With Llama2')
input_text = st.text_input("Search the topic you want")

# ollama LLAma2 LLm 
llm=Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
# chain = prompt | llm 

if input_text:
    # st.write(chain.invoke({'question': input_text}))
    st.write(chain.invoke({'question': input_text}))
