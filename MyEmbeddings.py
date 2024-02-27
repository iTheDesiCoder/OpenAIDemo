import os
import sys
from langchain.vectorstores.chroma import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from MyConfig import OPENAI_API_KEY

# OpenAI API settings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-3-large")
openai = OpenAI(openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-instruct")

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
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=512, chunk_overlap=20,length_function = len)
    texts = text_splitter.split_text(raw_text)
    if PERSIST:
        db = Chroma.from_texts(texts=texts, persist_directory=persist_directory, embedding=embedding)
        
    else:
        db = Chroma.from_texts(texts=texts, embedding=embedding)
        

# query it
query = "Can you explain about Fees and expenses"
print("\nQUESTION: \"%s\"" % query)
print("ANSWER: \n")
answers = db.similarity_search(query)
for answer in answers:
    print("%s\n" % answer.page_content)