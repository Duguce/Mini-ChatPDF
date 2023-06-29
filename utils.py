# -*- coding: utf-8  -*-
# @Author  : Duguce 
# @Email   : zhgyqc@163.com
# @Time    : 2023/6/10 19:43
# @File    : utils.py
# @Software: PyCharm
import os
import sys
import json

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
    """
    Select the files
    """
    cls()
    files = [file for file in os.listdir(FILES) if file.endswith('.pdf')]
    if not files:
        print("No PDF files in the folder")
        return None
    print("📁 Please select the file to process: ")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    print()

    try:
        possible_selections = list(range(len(files) + 1))

        selections = input("Please enter the file number, separate multiple files with spaces, or enter 0 to exit: ").split()
        selections = [int(selection) for selection in selections]

        if not set(selections).issubset(set(possible_selections)):
            return select_files()
        elif 0 in selections:
            handle_exit()
        else:
            pdfs_path = [os.path.join(FILES, files[i - 1]) for i in selections]
            return pdfs_path
    except ValueError:
        return select_files()


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

