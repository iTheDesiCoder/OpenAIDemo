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

# Read the PDF file and extract the text
pdf_reader = PdfReader("DataPDF/product.pdf")
raw_text = ''
for i, page in enumerate(pdf_reader.pages):
    content = page.extract_text()
    raw_text += content

text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(raw_text)

for j,text in enumerate(texts):
    print("%s\n" % text)
