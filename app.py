"""
El Clásico AI — Flask application entry point.
Handles routing, RAG pipeline orchestration, and chat API.
"""
import logging
import os
import sys
import uuid

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, save_message, get_history
from rag.vector_store import build_index, ingest_new_chunks
from rag.retrieval import retrieve, format_context
from rag.llm import ask
from rag.file_parser import parse_file, validate_file, is_supported_file
from rag.file_manager import save_uploaded_file, register_upload
from rag.ingestion import ingest_document, merge_with_existing

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "clasico-secret-key-2024")

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Global RAG state — loaded once at startup
_index = None
_chunks = None
_sources = None


def get_rag():
    global _index, _chunks, _sources
    # Always refresh the index from disk if source files changed.
    # build_index() will load cached data or rebuild when needed.
    _index, _chunks, _sources = build_index(DATA_FOLDER)
    return _index, _chunks, _sources


@app.route("/")
def index():
    # Assign a persistent session ID
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    session["session_id"] = session_id

    try:
        # 1. Load RAG index
        idx, chunks, sources = get_rag()

        # 2. Retrieve relevant context
        results = retrieve(question, idx, chunks, sources)
        context = format_context(results)

        # 3. Load conversation history for memory
        history = get_history(session_id, limit=10)
        llm_history = [{"role": h["role"], "content": h["content"]} for h in history]

        # 4. Generate answer
        answer = ask(question, context, llm_history)

        # 5. Persist this turn
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", answer)

        return jsonify({
            "answer": answer,
            "sources_used": len(results) > 0,
            "session_id": session_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def history():
    session_id = session.get("session_id")
    if not session_id:
        return jsonify([])
    return jsonify(get_history(session_id))


@app.route("/api/new_session", methods=["POST"])
def new_session():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"session_id": session["session_id"]})


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Upload a document (.txt, .pdf, .docx) and ingest it into the RAG system.
    File is automatically chunked, embedded, and added to the FAISS index.
    """
    global _index, _chunks, _sources
    
    try:
        # 1. Validate file upload
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not is_supported_file(file.filename):
            return jsonify({"error": "File type not supported. Use: .txt, .pdf, .docx"}), 400
        
        # 2. Save uploaded file
        file_path = save_uploaded_file(file, file.filename)
        
        # 3. Validate file
        is_valid, error_msg = validate_file(file_path, max_size_mb=50)
        if not is_valid:
            os.remove(file_path)
            return jsonify({"error": error_msg}), 400
        
        logging.info(f"Processing upload: {file.filename}")
        
        # 4. Parse file to extract text
        try:
            text = parse_file(file_path)
        except Exception as e:
            os.remove(file_path)
            logging.error(f"Failed to parse file: {e}")
            return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400
        
        # 5. Ensure RAG index is loaded
        if _index is None:
            index, chunks, sources = build_index(DATA_FOLDER)
            _index, _chunks, _sources = index, chunks, sources
        
        # 6. Ingest document (chunk and prepare for embedding)
        new_chunks, new_sources = ingest_document(
            text,
            source_name=file.filename,
            chunk_size=500,
            overlap=100
        )
        
        # 7. Add to existing chunks and avoid duplicates
        deduped_chunks, deduped_sources, merged_chunks, merged_sources = merge_with_existing(
            new_chunks, new_sources,
            _chunks, _sources
        )
        
        # 8. Ingest only truly new chunks into FAISS index
        _index, _chunks, _sources = ingest_new_chunks(
            _index,
            _chunks, _sources,
            deduped_chunks, deduped_sources
        )
        
        if deduped_chunks:
            _chunks, _sources = merged_chunks, merged_sources

        # 9. Register in upload metadata
        register_upload(file.filename, file_path, len(deduped_chunks))
        
        logging.info(f"Successfully ingested: {file.filename} ({len(deduped_chunks)} new chunks added)")
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "chunks_added": len(deduped_chunks),
            "message": f"Document ingested: {file.filename} ({len(deduped_chunks)} chunks added)"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    print("Initializing RAG index...")
    get_rag()
    print("El Clásico AI ready at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
