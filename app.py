# libraries
import numpy as np
from flask import Flask, render_template, request
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import pandas as pd


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Document(metadata={self.metadata}, page_content='{self.page_content}')"



# Webserver instantiatin
app = Flask(__name__)



# Openai key
os.environ['OPENAI_API_KEY'] = ""

# add all files needed from the folder
allDocuments = []
for file in os.listdir("./data/"):
    file_path = os.path.join("./data/", file)
    if file.lower().endswith('.pdf'):
        # Process PDF files
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        allDocuments.extend(documents)
    elif file.lower().endswith('.csv'):
        # Process CSV files
        df = pd.read_csv(file_path)
        docs = df.to_string(index=False)
        documents = [Document(row, metadata={'source': file_path, 'page': i}) for i, row in enumerate(docs.split('\n'))]
        print(len(documents))
        allDocuments.extend(documents)


# split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(allDocuments)


# define embedding
embeddings = OpenAIEmbeddings()
# create vector database from data
db = DocArrayInMemorySearch.from_documents(docs, embeddings)
# define retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20, 'fetch_k': 10000})
# retriever = db.as_retriever( search_type="mmr",
#         search_kwargs={'k': 5, 'fetch_k': 1000})

# create a chatbot chain. Memory is managed externally.
qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0), 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    return_generated_question=True,
)



# Questions and answers for recording
questions = [
    # "Name",
    # "Gender",
    # "Health Issue",
    # "Request",
]
answers = []
index = 0
chat_history = []



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    global index
    user_msg = request.form["msg"]
    answers.append(user_msg)
    if index < len(questions):
        query = f"My {questions[index]} {user_msg}"
        result = qa({"question": user_msg, "chat_history": chat_history})
        chat_history.extend([(query, result["answer"])])
        index += 1
        if index != len(questions):
            return "What is your " + questions[index] + " ?"
        result = qa({"question": user_msg, "chat_history": chat_history})
        chat_history.extend([(user_msg, result["answer"])])
        answer = result['answer'] 
        return  answer


    else:
        result = qa({"question": user_msg, "chat_history": chat_history})
        chat_history.extend([(user_msg, result["answer"])])
        answer = result['answer'] 
        return  answer


if __name__ == "__main__":
    app.run()