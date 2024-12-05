from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": ["http://127.0.0.1:8000", "http://localhost:8000"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})


CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
The following is a conversation between a user and an assistant.

{history}

Now, based on the following context, answer the user's last question.

Context:
{context}

---

Assistant, please answer the user's question based on the above context: {question}
"""


embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = OllamaLLM(model="llama3.2:latest", streaming=True) # enable streaming gpt-like

@app.route('/api/query', methods=['POST', 'OPTIONS'])
def api_query():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response

    data = request.get_json()
    query_text = data.get('query')
    conversation_history = data.get('conversation_history', [])

    def generate():
        for chunk in query_rag_stream(query_text, conversation_history):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')

def query_rag_stream(query_text: str, conversation_history):
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    history_text = ""
    for speaker, text in conversation_history:
        history_text += f"{speaker}: {text}\n"

    # create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, history=history_text)

    for chunk in model.stream(prompt):
        yield chunk  # yield each chunk as it's generated

if __name__ == "__main__":
    app.run(debug=True, port=5001)
