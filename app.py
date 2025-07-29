from langchain_groq import ChatGroq
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
import dotenv
from dotenv import load_dotenv
load_dotenv()


import asyncio

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# api key 
perplexity_api_key=os.getenv("PERPLEXITY_API_KEY")
google_api_key=os.getenv("GOOGLE_API_KEY")

# llm=ChatOpenAI(
#     api_key=perplexity_api_key,
#     base_url="https://api.perplexity.ai",
#     model="sonar-pro"
# )

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


st.title("converstaional tag with pdf upload and chat history ")
st.write("upload pdf and chat with content")

api_key=perplexity_api_key

# check if api key provided

if api_key:
    llm=ChatOpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai",
        model="sonar-pro"
    )
    
    session_id=st.text_input("SESSION_ID" , value="deafult_Session")

    # statefully manaage chat hostory 
    if 'store' not in st.session_state:
        st.session_state.store={}    # store all messages 


uploaded_files=st.file_uploader("choose a pdf file" , type="pdf" , accept_multiple_files=True)

# process uploaded_files

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temp_pdf=f"./temp.pdf"
        with open(temp_pdf , "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name

        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)


    # // create embedding and split 

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000 , chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vector_store=Chroma.from_documents(documents=splits , embedding=embeddings)
    retriever=vector_store.as_retriever()

    
    cont_q_system_prompt=(
        "given the chat history and latest user question "
        "which might referecne context in chat history"
        "formalate a standalone question which can be understood"
        "without the chat history . do not answer question , "
        "just reformaulte it if needed otherwise return it as it is"
    )

    cont_q_promp=ChatPromptTemplate.from_messages(
        [
            ("system" , cont_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human" , '{input}')
        ]
    )

    history_aware_retriver=create_history_aware_retriever(llm , retriever , cont_q_promp)

    # answer  question 

    system_prompt= (
        """
        you are an ai assistant for question-answer task use the following piece of retrived information
        to answer thr question if you dont know answer say that you dont know use three sentence max to keep 
        answer concise
        \n\n
        {context}
    """
    )

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system" , system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human" , '{input}')
        ]
    )

    question_answer_Chain=create_stuff_documents_chain(llm , prompt=prompt)
    rag_chain=create_retrieval_chain(history_aware_retriver , question_answer_Chain)


    def get_session_history(session_id)->BaseChatMessageHistory :
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()

        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


    user_input=st.text_input("your question")
    if user_input:
        session_history=get_session_history(session_id)
        # response=conversational_rag_chain(
        #     {
        #         "input" : user_input
        #     },
        #     config={
        #         "configurable" : {
        #             "session_id" : session_id
        #         }
        #     }
        # )
        response = asyncio.run(conversational_rag_chain.ainvoke(
            {
                "input": user_input
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        ))

        st.write(st.session_state.store)
        st.success(f"Assistant: {response['answer']}")
        st.write("chat hostory : " , session_history.messages)

# else:
#     st.warning("please give api key")