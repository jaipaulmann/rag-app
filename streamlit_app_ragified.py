from openai import OpenAI
import streamlit as st
import boto3
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time

import os
from dotenv import load_dotenv


# import secrets
def configure():
    load_dotenv()

configure()

kb_id = os.getenv('kb_id') #calling the environment variable
access_key = os.getenv('aws_access_key_id') 
secret_access_key = os.getenv('aws_secret_access_key') 

# set prompt template
Prompt_template = """
    Human: The given context come from the engineering/technical blogs or news from the company website.
    We are making an assumption that these blogs are up-to-date and relevant to the latest AI projects the company is working on.
     
    Given the question and context below, you will summarize the context and answer the question.
    
    <context>
    {context}
    </context>
    
    <question>
    {input}
    </question>
    
    Assistant:"""
    
# define pipeline, supply credentials needed
def pipeline_setup(kb_id, access_key, secret_access_key):
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_access_key,
        region_name="us-west-2"
    )

    # configure model details
    model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.5, "top_p": 0.9}
    modelID = "anthropic.claude-v2"

    llm = BedrockLLM(
        model_id=modelID,
        client=bedrock_client,
        model_kwargs=model_kwargs
    )

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 3}},
        region_name="us-west-2",
    )

    claude_prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=Prompt_template
    )
    
    combine_docs_chain = create_stuff_documents_chain(
                llm, claude_prompt
            )
    
    pipeline = create_retrieval_chain(retriever, combine_docs_chain)
    return pipeline 


# define function to produce response from our Bedrock + Knowledge Base 
    # provide query + pipeline -> receive response
def invoke(query, pipeline):
    answer = pipeline.invoke({'input': query})
    return answer

# UI: set title of application
st.title("CurrentAI Rag Chatbot")
st.logo("current-ai-logo.png")

# check if there are any messages in state (memory)
    # if nothing, start a new list object where we will store all messages back and forth
if "messages" not in st.session_state:
    st.session_state.messages = []

# UI: loop through the list of all messages,  and render them in the chat UI
    # ensure that messages are tied to the appropriate role, ie. User or Assistant
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# instantiate pipeline
pipeline = pipeline_setup(kb_id, access_key, secret_access_key)
query = "How does IBM use machine learning lately?"
answer = invoke(query, pipeline)
print(answer)


# check if a new prompt has been recieved via the Chat Input box
        # store the user's question in variable "prompt"
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt}) # append new prompt to history

    with st.chat_message("user"): # render new prompt in the Chat UI w/ "user" icon
        st.markdown(prompt) 

    with st.chat_message("assistant"): # render new prompt in the Chat UI w/ "Assistant" icon
        query = prompt # set query variable equal to the user prompt that was just received via chat input
        answer = invoke(query, pipeline) # sending query via pipeline to bedrock, get a response

        stream = answer['answer'] # placeholder incase we decide to send answer through generator for typewriter effect
        sources_cited = answer['context'][0].metadata['source_metadata']['url'] # store the source of answer!

        # Concatenate the strings with desired formatting
        output_text = f"{stream}\n\nSources Cited:\n{sources_cited}"

        def stream_data():
            for word in output_text.split(" "):
                yield word + " "
                time.sleep(0.02)
                
        response = st.write(stream_data) # outputs the final output w/ Assistant Icon

    st.session_state.messages.append({"role": "assistant", "content": response}) # adds response to history!








##### Rebuilding for our RAG use above
#############################################################################
    
#############################################################################
    
#############################################################################

# # set title of application
# st.title("CurrentAI Rag Chatbot")
# st.logo("current-ai-logo.png")

# # bring in environment variables
# client = OpenAI(api_key="sk-my-service-account-8bgNv3BMZq07yHaNNdhMT3BlbkFJuBjO46gETHEzogokcsWA")

# # define any params needed for app to work
#     # open ai requires a model specification (below)
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # check if there are any messages in state (memory)
#     # if nothing, start a new list object where we will store all messages back and forth
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # loop through the list of all messages,  and render them in the chat UI
#     # ensure that messages are tied to the appropriate role, ie. User or Assistant
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # check if a new prompt has been recieved via the Chat Input box
#         # store the user's question in variable "prompt"
# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt}) # append new prompt to history

#     with st.chat_message("user"): # render new prompt in the Chat UI w/ "user" icon
#         st.markdown(prompt) 

#     with st.chat_message("assistant"): # render new prompt in the Chat UI w/ "Assistant" icon
#         stream = client.chat.completions.create( # hit OpenAI's completion API endpoint
#             model=st.session_state["openai_model"], # pass it the openai model value from above
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages # pass in entire message history as list of dicts
#             ],
#             stream=True, # openai supports streaming, renders as typewriter effect
#         ) # hit OpenAI's completion API endpoint
#         response = st.write_stream(stream) # outputs the final output w/ Assistant Icon

#     st.session_state.messages.append({"role": "assistant", "content": response}) # adds response to history!