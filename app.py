from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
import re


app = Flask(__name__)
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
index_name = "medibot-index"

# Load embeddings and vector store
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 5})

# Load HuggingFace model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set up the text2text generation pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Wrap in LangChain LLM interface
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Prompt + Retrieval-Augmented Generation Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def clean_context(text):
    # Remove lines that look like chapter titles, numbers, or page references
    text = re.sub(r'\bCHAPTER\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsymptoms\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfigure\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btable\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,3}\s+[A-Za-z ,\-]+CHAPTER\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)  # remove newlines
    text = re.sub(r'\s+', ' ', text)  # normalize all spaces
    return text.strip()


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    try:
        # Greeting handler
        normalized_input = msg.strip().lower()
        GREETINGS = {
            "hi": "Hello! How can I assist you today?",
            "hello": "Hi there! What medical question can I help with?",
            "how are you": "I'm just a helpful bot, always ready to assist you with medical info!",
            "thank you": "You're welcome! Let me know if you have more questions.",
            "thanks": "Glad I could help!"
        }
        for greeting in GREETINGS:
            if greeting in normalized_input:
                return GREETINGS[greeting]

        # Retrieve context
        retrieved_docs = retriever.get_relevant_documents(msg)
        context = "\n".join([clean_context(doc.page_content) for doc in retrieved_docs])

        # Fallback if no context
        if not context.strip():
            return "Sorry, I don't have enough information to answer this."

        # Check if context contains relevant medical info
        if not any(keyword in context.lower() for keyword in ["cause", "treatment", "medication", "precaution"]):
            return "Sorry, I don't have enough information to answer this."

        # Split context if too long
        MAX_TOKENS = 512
        words = context.split()
        if len(words) > MAX_TOKENS:
            context_chunks = [" ".join(words[i:i + MAX_TOKENS]) for i in range(0, len(words), MAX_TOKENS)]
        else:
            context_chunks = [context]

        # Generate responses
        responses = []
        for chunk in context_chunks:
            formatted_prompt = prompt.format(input=msg, context=chunk)
            response = llm(formatted_prompt)
            responses.append(response)

        final_response = " ".join(responses).replace("Human:", "").strip()
        return final_response

    except Exception as e:
        print("Error:", e)
        return "\u26a0\ufe0f Error processing your request. Please try again."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)