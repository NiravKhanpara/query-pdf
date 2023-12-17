from dotenv import load_dotenv
load_dotenv()

import os
import pickle
import streamlit as st
from scanned_pdf_parser import get_text_from_scanned_pdf
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

llm = GooglePalm(temperature=0.9)

st.title("PDF Query Tool")
st.write("Upload your PDF and ask question from it")

uploaded_file = st.file_uploader("Choose a PDF file")
main_placeholder = st.empty()
second_placeholder = st.empty()


if uploaded_file:
    filename = uploaded_file.name
    if not filename.endswith(('.pdf', '.PDF')):
        main_placeholder.warning("Choose PDF Document !!!")
        exit()
    elif not os.path.exists(uploaded_file.name):
        main_placeholder.text("Data Loading Started...⌛⌛⌛")
        with open(f'{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.getbuffer())

        pdf_loader = PyPDFLoader(uploaded_file.name)
        documents = pdf_loader.load()

        raw_text = ''
        for doc in documents:
            raw_text += doc.page_content

        if len(raw_text) < 10:
            main_placeholder.text("It looks like Scanned PDF, No worries converting it...⌛⌛⌛")
            raw_text = get_text_from_scanned_pdf(uploaded_file.name)

        main_placeholder.text("Splitting text into smaller chunks...⌛⌛⌛")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=2000
        )

        texts = text_splitter.split_text(raw_text)
        docs = [Document(page_content=t) for t in texts]

        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
        main_placeholder.text("Storing data into Vector Database...⌛⌛⌛")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save the FAISS index to a pickle file
        with open(f'vector_store_{uploaded_file.name}.pkl', "wb") as f:
            pickle.dump(vectorstore, f)

    main_placeholder.text("Data Loading Completed...✅✅✅")


query = second_placeholder.text_input("Question:")
if query:
    if os.path.exists(f'vector_store_{uploaded_file.name}.pkl'):
        with open(f'vector_store_{uploaded_file.name}.pkl', "rb") as f:
            vector_store = pickle.load(f)

        prompt_template = """
            <context>
            {context}
            </context>
            Question: {question}
            Assistant:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        with st.spinner("Searching for the answer..."):
            result = chain({"query": query})
        st.header("Answer")
        st.write(result["result"])

