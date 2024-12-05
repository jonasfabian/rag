from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

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

def main():
    conversation_history = []
    print("Type 'exit' to quit.")
    while True:
        query_text = input("You: ")
        if query_text.lower() == "exit":
            break
        response = query_rag(query_text, conversation_history)
        print(f"Assistant: {response}")
        conversation_history.append(("User", query_text))
        conversation_history.append(("Assistant", response))

def query_rag(query_text: str, conversation_history):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    history_text = ""
    for speaker, text in conversation_history:
        history_text += f"{speaker}: {text}\n"

    # create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, history=history_text)

    model = OllamaLLM(model="llama3.2-vision:latest")
    response_text = model.invoke(prompt)

    return response_text

if __name__ == "__main__":
    main()
