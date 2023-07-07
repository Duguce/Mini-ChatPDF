
# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/6/10 19:43
# @File    : utils.py
# @Software: PyCharm
import os
import sys
import json
import streamlit as st

# PDF files path
FILES = os.path.join(os.getcwd(), 'pdf_files')


def initialize():
    """
    Initialize the folder
    """
    if not os.path.exists(FILES):
        os.makedirs(FILES)


def cls():
    """
    Clear the screen
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def select_files():
    st.title("PDF Q&A Assistant")
    st.subheader("Please upload the PDFs you'd like to process")
    uploaded_files = st.file_uploader("upload PDF files", accept_multiple_files=True, type=['pdf'], label_visibility='hidden')

    if not uploaded_files:
        st.warning("Please upload at least one PDF file to proceed.")
        return None

    st.markdown("""---""")

    os.makedirs(FILES, exist_ok=True)

    pdfs_path = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(FILES, uploaded_file.name)
        pdfs_path.append(file_path)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return pdfs_path


def handle_exit():
    """
    Exit the program
    """
    print("🤖 Chatbot: Bye bye!\n")
    sys.exit(1)


def handle_save(title, history):
    """
    Save the history
    """
    with open(f"{title}.json", "w") as f:
        json.dump(history, f)

    print(f"📝 Save successful! Filename: {title}.json\n")

