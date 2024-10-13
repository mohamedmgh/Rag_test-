
import random
import re
import subprocess
import time
import urllib
import warnings
from ast import Global
from pathlib import Path
from pathlib import Path as p
from pprint import pprint
from statistics import fmean
from typing import Any

import mlflow
import pandas as pd
import pdfplumber
import tabula
import vertexai
import vertexai.preview.generative_models as generative_models
from genair_core.embeddings import CustomVertexAIEmbeddings
from IPython.core.interactiveshell import InteractiveShell
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel
from mlflow.data.pandas_dataset import PandasDataset
from numpy import integer
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Part,
)

InteractiveShell.ast_node_interactivity = "all"
PROJECT_ID = "prj-8cf1-genaipov-dev-9744"  # @param {type:"string"}
LOCATION = "europe-west4"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION,api_transport="rest")


warnings.simplefilter(action='ignore', category=FutureWarning)


#                                **************************************          PDF Loading and goal standard                * **************************************  # noqa: E501


data_path = Path().resolve().parent.parent.parent / "data"



#Read Table page 41 and page 42 and page38 and 43



#                                **************************************         LLM initialization                    * **************************************  # noqa: E501

#MODEL_ID = "gemini-1.5-pro-001"  # @param {type:"string"}
MODEL_ID = "gemini-1.5-flash-001"
model = GenerativeModel(MODEL_ID)
# Load a example model with system instructions
example_model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a manufacturing engineer that provides torque values",

    ],
)
# Set model parameters
generation_config = {
    "temperature":0.9,
    "top_p":1.0,
    "top_k":32,
    "candidate_count":1,
    "max_output_tokens":8192,
}


# Set safety settings

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}





# Create a list of indices from start_index to end_index
iindex_list = list(sampple_values.index)

#Extract tables from PDF files using pdf_plumber
pdf_plumber = pdfplumber.open(pdf_path)
liste_page=[37,40,41,42]
extracted_table=[]
for i in (liste_page):
    print(pdf_plumber.pages[i].extract_table())
    extracted_table = extracted_table+pdf_plumber.pages[i].extract_tables()

print("---------------------------------------------------------------------------------------------------------------s")
extraction_method=extracted_table
print(extraction_method)

with open( data_path / "cptr_page_41.jpeg", "rb") as file:
    image1 = Part.from_data(
    mime_type="image/jpeg",
    data=file.read()
)

with open( data_path / "cptr_page_42.jpeg", "rb") as file:
    image2 = Part.from_data(
  mime_type="image/jpeg",
    data=file.read()
)

with open( data_path / "Cptr_page_38.jpeg", "rb") as file:
    image3 = Part.from_data(
                mime_type="image/jpeg",
                data=file.read()
        )

with open( data_path / "cptr_page_43.jpeg", "rb") as file:
 image4 = Part.from_data(
 mime_type="image/jpeg",
 data=file.read()
)




warnings.filterwarnings("ignore")
# restart python kernal if issues with langchain import.



pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load_and_split()
print(pages[42].page_content)

#      --------------------       ------------------   RAG Pipeline: Embedding + Gemini (LLM)



text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n"],
    chunk_size = 5000,
    chunk_overlap  = 0,
    length_function = len,
)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

# Import CustomVertexAIEmbeddings which retries failed API requests with exponential backoff.
from genair_core.embeddings import CustomVertexAIEmbeddings
vertexai.init(
    api_transport="rest",
    project="prj-8cf1-genaipov-dev-9744",
    location="europe-west3",
)


# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 4
embedding_model = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
    model_name="textembedding-gecko@002"
)



from genair_core.retrievers import (s
    InMemoryKeywordSearch,
)
from genair_core.retrievers import (
    InMemorySemanticSearch,
)
from langchain_core.documents import Document
documents = [Document(
    page_content=texts[i],
   metadata={"product_id": ["abp"]}
)for i in range(len(texts))]

keyword_retriever = InMemoryKeywordSearch.from_documents(
    documents=documents,
    max_results=3
)
search_text ='NSA5474'
results = keyword_retriever.invoke(search_text)

for i, output in enumerate(results):
    print(f"Nearest neighbor {i}: {output}")
    print("")

