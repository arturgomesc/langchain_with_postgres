from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

load_dotenv() # carregar a API KEY da OpenAI 

loader = TextLoader('universal_declaration_of_human_rights.txt', encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents) # Separar o documento em partes

embeddings = OpenAIEmbeddings()

CONNECTION_STRING = "postgresql+psycopg2://postgres:7834@localhost:5432/vector_db2" # conexão com o Postgresql
COLLECTION_NAME = 'universal_declaration_of_human_rights_vectors'

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
) # convertendo as partes do documento em embeddings e adicionando ao Postgresql com PGVector

query = "What did ONU said about justice"

similar = db.similarity_search_with_score(query, k=2)

for doc in similar:
    print(doc)
