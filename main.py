# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/6/10 20:22
# @File    : main.py
# @Software: PyCharm
import os
import datetime
import fitz
import openai
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from openai.embeddings_utils import distances_from_embeddings
from utils import cls, handle_save, handle_exit, initialize, select_files
import streamlit as st

# Load the environment variables
load_dotenv()
 
# openai.api_type = os.environ["OPENAI_API_TYPE"]
# openai.api_version = os.environ["OPENAI_API_VERSION"]

# The models to use
models = {
    "gpt-4": "gpt-4",
    "embeddings": "text-embedding-ada-002"
}

# The history of the conversation
history = []

# Load the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# The maximum number of tokens to generate
MAX_TOKENS = 500


def extract_text(pdfs_path):
    """
    Extract the text from the PDF files
    """
    cls()
    print("📄 Extracting text...")

    # Create a list to store the extracted text
    extracted_text = []
    # Create a list to store the PDF numbers
    pdfs_num = []
    for pdf_num, pdf_path in enumerate(pdfs_path, start=1):
        # Open the PDF file
        doc = fitz.open(pdf_path)
        # Extract the text from each page of the PDF file
        text = " ".join(page.get_text() for page in doc)
        # Append the extracted text to the list
        extracted_text.append(text)
        # Append the PDF number to the list
        pdfs_num.append(pdf_num)

    # Create a DataFrame from the extracted text list
    text_df = pd.DataFrame({"text": extracted_text})
    # Add the PDF numbers to the DataFrame
    text_df["no_pdf"] = pdfs_num

    return text_df


def split_2_sentences(text, max_tokens=MAX_TOKENS):
    """
    Split the text into sentences
    """
    sentences = text.split(". ")  # Split the text into sentences
    n_tokens = [len(tokenizer.encode(f" {sentence}")) for sentence in
                sentences]  # Get the number of tokens for each sentence

    chunks = []  # Create a list to store the chunks
    tokens_so_far = 0  # Initialize the number of tokens so far
    chunk = []  # Create a list to store the sentences of the current chunk

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ". ")  # Append the chunk to the list of chunks
            chunk = []  # Reset the chunk
            tokens_so_far = 0  # Reset the number of tokens so far

        if token > max_tokens:
            continue

        chunk.append(sentence)  # Append the sentence to the chunk
        tokens_so_far += token + 1  # Update the number of tokens so far

    return chunks


def create_embeddings(text_df, model=models["embeddings"]):
    """
    Create the embeddings for the text
    """
    cls()
    print("🔍 Generating embedding vectors...")
    # Add the number of tokens to the DataFrame
    text_df["n_tokens"] = text_df["text"].apply(lambda x: len(tokenizer.encode(x)))
    # Create a list to store the shortened text
    shortened = []
    # Create a list to store the PDF numbers
    no_pdf = []
    # Iterate over the rows of the DataFrame
    for row in text_df.iterrows():

        if row[1]['text'] is None:
            continue

        if row[1]['n_tokens'] > MAX_TOKENS:
            chunks = split_2_sentences(row[1]['text'])
            shortened.extend(chunks)
        else:
            shortened.append(row[1]['text'])

        no_pdf.extend([row[1]['no_pdf']] * len(chunks))

    # Create a DataFrame from the shortened text list
    embeddings_df = pd.DataFrame(shortened, columns=["text"])
    # Add the PDF numbers to the DataFrame
    embeddings_df["no_pdf"] = no_pdf
    embeddings_df["n_tokens"] = embeddings_df["text"].apply(lambda x: len(tokenizer.encode(x)))
    embeddings_df["embedding"] = embeddings_df["text"].apply(lambda x: openai.Embedding.create(
        input=x, engine=model)['data'][0]['embedding'])
    # Reorder the columns of the DataFrame
    embeddings_df = embeddings_df[['no_pdf', 'text', 'n_tokens', 'embedding']]

    return embeddings_df


def cal_similarity(question, embeddings_df, model=models["embeddings"], max_len=1800):
    """
    Calculate the similarity between the question and the text (return the most similar text)
    """
    # Create a list to store the similarity scores
    similarity_df = pd.DataFrame()
    # Create the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question,
                                           engine=model)['data'][0]['embedding']
    # Calculate the distance between the question and the text
    similarity_df["distance"] = distances_from_embeddings(q_embeddings,
                                                          embeddings_df["embedding"].values,
                                                          distance_metric="cosine")
    # Add the PDF numbers to the DataFrame
    similarity_df["no_pdf"] = embeddings_df["no_pdf"].values
    # add text
    similarity_df["text"] = embeddings_df["text"].values
    # add n_tokens
    similarity_df["n_tokens"] = embeddings_df["n_tokens"].values
    # add text embedding
    similarity_df["embedding"] = embeddings_df["embedding"].values
    # Sort the DataFrame by the distance
    similarity_df.sort_values(by="distance", inplace=True)

    context = []  # Create a list to store the context
    curr_len = 0  # Initialize the current length
    for _, row in similarity_df.iterrows():
        curr_len += row["n_tokens"] + 4  # Update the current length
        if curr_len > max_len:
            break
        # Append the text to the context
        context.append(row["text"])

    return "\n\n###\n\n".join(context), similarity_df

def process_single_pdf(file_path):
    # Extract file name without extension
    file_name = os.path.basename(file_path).split(".")[0]
    file_dir = os.path.dirname(file_path)

    # Look for the processed embedding file in the folder
    embedding_file_path = os.path.join(file_dir, f"{file_name}_embed.pkl")
    if os.path.exists(embedding_file_path):
        # if found, load the file
        embeddings_df = pd.read_pickle(embedding_file_path)
        return embeddings_df

    # If not found, process the file: extract text, create embeddings, and save
    text_df = extract_text([file_path])
    embeddings_df = create_embeddings(text_df)
    embeddings_df.to_pickle(embedding_file_path)
    return embeddings_df

def chat(pdfs_path, model=models["gpt-4"]):
    if not pdfs_path:
        return

    pdfs_embed = [process_single_pdf(pdf_path) for pdf_path in pdfs_path]
    embeddings = pd.concat(pdfs_embed)
    st.subheader("Ask the Chatbot About Your PDFs")
    user_input = st.text_input(label="Enter your question:")
    if user_input:
        context, _ = cal_similarity(user_input, embeddings)
        user_message = {"role": "user", "content": f"The pdf content:{context}, the question is: {user_input}"}
        history.append({"role": "user", "content": user_input})
        tmp_message = []
        tmp_message.extend(history)
        tmp_message.append(user_message)
        res = openai.ChatCompletion.create(
            engine=model,
            model=model,
            messages=tmp_message,
        )['choices'][0]["message"]["content"]
        st.info(f"{res}")
        history.append({"role": "assistant", "content": res})

def initialize_streamlit():
    cls()
    print("🌟 PDF Q&A Assistant 🌟")
    print("\nYou can ask questions about the PDFs you provided!")
    print("\nPowered by OpenAI GPT-4 Language Model")


def main():
    initialize_streamlit()
    pdfs_path = select_files()

    with st.sidebar:
        st.title("PDF Q&A Assistant Settings")
        st.markdown("""
            This tool allows you to ask questions about the content of your PDF files.
            It uses the OpenAI GPT-4 Language Model to search for answers within your documents.
        """)

        openai.api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        openai.api_base = st.text_input("Enter your OpenAI API Base URL (optional):",
                                         "https://api.openai.com")

        global MAX_TOKENS
        MAX_TOKENS = st.number_input("Enter the Max Tokens (default is 500):", value=500, step=1)

    if openai.api_key:
        chat(pdfs_path)
    else:
        st.warning("Please enter your OpenAI API Key in the sidebar to proceed.")


if __name__ == '__main__':
    main()
