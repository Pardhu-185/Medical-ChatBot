from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted_data = load_pdf(data="data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# --- Initialize Pinecone and Create Index ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medibot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)

# Create an instance of the Pinecone Index through the Pinecone client
from langchain_pinecone import PineconeVectorStore

# Initialize PineconeVectorStore wrapper
docsearch = PineconeVectorStore(
    index=pc.Index(index_name),  # ✅ uses new v3+ Index object
    embedding=embeddings
)

# Upload documents in smaller batches to avoid payload size errors
batch_size = 25
for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    try:
        docsearch.add_documents(batch)
        print(f"✅ Uploaded batch {i // batch_size + 1}")
    except Exception as e:
        print(f"❌ Failed batch {i // batch_size + 1}: {e}")