import os
from openai import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)

loader = TextLoader('thomas_sankara_speech.txt', encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
# print(len(texts)) -> 5

embeddings = OpenAIEmbeddings()
doc_vector = embeddings.embed_documents([t.page_content for t in texts])

CONNECTION_STRING = "postgresql+psycopg2://postgres:7834@localhost:5432/vector_db"
COLLECTION_NAME = 'thomas_sankara_speech'

db = PGVector.from_documents(embedding = embeddings, documents=texts, collection_name = COLLECTION_NAME,
              connection_string = CONNECTION_STRING)