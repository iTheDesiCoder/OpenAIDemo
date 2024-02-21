import os
import sys
from langchain.vectorstores.chroma import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from datasets import load_dataset
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from MyConfig import OPENAI_API_KEY

# OpenAI API settings

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-4-turbo-preview")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-large")

# Create a vectorstore
PERSIST = True
persist_directory = "persist"
if PERSIST and os.path.exists(persist_directory):
    print("Reusing index...\n")
    db = Chroma(persist_directory=persist_directory,embedding_function=embedding)
else:
    # Read the PDF file and extract the text
    pdf_reader = PdfReader("DataPDF/product.pdf")
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    #print(raw_text)
        
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=20,length_function = len)
    texts = text_splitter.split_text(raw_text)
    if PERSIST:
        db = Chroma.from_texts(texts=texts, persist_directory=persist_directory, embedding=embedding)
    else:
        db = Chroma.from_texts(texts=texts, embedding=embedding)

# query it
query = "Can you list funds which has lowest fees"

answer = db.similarity_search(query, k=1)



system_message = "You are a helpful Annuity sales assistant who is always available to answer questions about the product to Financial Advisor. You are knowledgeable about the product and can answer questions about the product features, benefits, subaccounts, and how to use the product." 
human_message = query + " " + answer[0].page_content
messages = [ HumanMessage(content=human_message), SystemMessage(content=system_message) ]

chat_response = chat.invoke(messages)
print("\nQUESTION: \"%s\"" % query)
print("ANSWER: %s\n" % chat_response.content)