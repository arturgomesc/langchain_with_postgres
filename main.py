from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('thomas_sankara_speech.txt', encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector = embeddings.embed_query('Testing the embedding model')

doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])

CONNECTION_STRING = "postgresql+psycopg2://postgres:7834@localhost:5432/vector_db2"
COLLECTION_NAME = 'thomas_sankara_speech_vectors'

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

query = "What did Thomas Sankara say about justice"

similar = db.similarity_search_with_score(query, k=2)

for doc in similar:
    print(doc)