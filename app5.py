import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pyprojroot import here

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for API keys
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing! Set it as an environment variable.")
if not GROQ_API_KEY:
    raise ValueError("Groq API Key is missing! Set it as an environment variable.")

# Constants
EMBEDDING_MODEL = "models/text-embedding-004"
VECTORDB_DIR = "vectordb"
COLLECTION_NAME = "chroma"
K = 2
TRANSCRIPT_FILE = "cleaned_transcript.txt"

# Bloom's Taxonomy levels
BLOOMS_TAXONOMY = {
    "remember": {"description": "Recall facts and basic concepts", "verbs": ["define", "list", "name", "identify", "recall"]},
    "understand": {"description": "Explain ideas or concepts", "verbs": ["explain", "describe", "summarize"]},
    "apply": {"description": "Use information in new situations", "verbs": ["apply", "solve", "demonstrate"]},
    "analyze": {"description": "Draw connections among ideas", "verbs": ["analyze", "differentiate"]},
    "evaluate": {"description": "Justify a decision", "verbs": ["evaluate", "assess"]},
    "create": {"description": "Produce new or original work", "verbs": ["create", "design"]}
}

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
show_chunks = False

# Load vectorstore
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(here(VECTORDB_DIR)),
        embedding_function=embeddings
    )
    return vectordb

# Generate custom prompt
def generate_prompt(cognitive_level, sentiment):
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Context: {context}\nQuestion: {question}\nAnswer: """
    )

# Get chat chain
def get_conversation_chain(vectorstore, memory, level="understand", sentiment="neutral"):
    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    prompt = generate_prompt(level, sentiment)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Detect level/sentiment (stub)
def detect_cognitive_level(text):
    return "apply" if "how" in text.lower() else "understand"

def detect_sentiment(text):
    return "neutral"

# Lookup context
def lookup_chunks(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=K)
    return [doc.page_content for doc in docs]

# Handle question
def handle_userinput(question, conversation):
    context = "\n\n".join(lookup_chunks(question, vectorstore))
    response = conversation({"question": question})
    return response.get("answer", ""), context

# Initialize conversation
vectorstore = get_vectorstore()
conversation = get_conversation_chain(vectorstore, memory)

# Chat Endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation, memory
    data = request.get_json()
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"error": "Missing question"}), 400
    try:
        answer, context = handle_userinput(question, conversation)
        return jsonify({
            "answer": answer,
            "context": context
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Question Generator Endpoint
@app.route("/api/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    course_outcome = data.get("course_outcome")
    bloom_level = data.get("bloom_level")

    if not course_outcome or not bloom_level:
        return jsonify({"error": "Missing course_outcome or bloom_level"}), 400

    prompt = f"""You are an expert question generator.\nBased on the following course outcome and Bloom level, generate:\n- One objective MCQ with 4 options (A-D)\n- One short answer subjective question\n\nCourse Outcome: {course_outcome}\nBloom Level: {bloom_level}\n\nFormat your response like this:\nObjective Question:\n...\nA. ...\nB. ...\nC. ...\nD. ...\n\nShort Answer Question:\n..."""

    model = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GOOGLE_API_KEY)
    result = model.invoke(prompt)
    content = result.content if hasattr(result, "content") else str(result)

    lines = content.strip().split('\n')
    options = [line for line in lines if line.strip().startswith(tuple("ABCD"))]
    subjective_start = next((i for i, l in enumerate(lines) if "Short Answer Question" in l), -1)
    subjective = lines[subjective_start + 1] if subjective_start >= 0 and subjective_start + 1 < len(lines) else "N/A"

    return jsonify({
        "bloom_level": bloom_level,
        "course_outcome": course_outcome,
        "questions": {
            "objective": {
                "question": "Here are the generated questions:",
                "options": options
            },
            "subjective": subjective
        },
        "raw_text": content
    })

# Clear conversation
@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    global conversation, memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    conversation = get_conversation_chain(vectorstore, memory)
    return jsonify({"message": "Conversation history cleared"})

# Toggle chunk visibility
@app.route('/api/toggle-chunks', methods=['POST'])
def toggle_chunks():
    global show_chunks
    data = request.get_json()
    val = data.get("show_chunks")
    if not isinstance(val, bool):
        return jsonify({"error": "Invalid value for show_chunks"}), 400
    show_chunks = val
    return jsonify({"message": f"Show chunks set to {show_chunks}", "show_chunks": show_chunks})

# Health Check
@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Dhamm AI backend is running."})

# Serve test UI
@app.route("/test")
def test_ui():
    return open("test.html", encoding="utf-8").read()

# Run Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
