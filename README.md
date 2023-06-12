English | [中文简体](https://github.com/Duguce/Mini-ChatPDF/blob/main/README.zh-CN.md)

# Mini-ChatPDF

This project is based on `GPT3.5-turbo` and can answer user's questions based on the PDF text files provided by the user.

Here is a simple example:

<img src="https://zhgyqc.oss-cn-hangzhou.aliyuncs.com/snipaste_20230612_164624.jpg" alt="效果展示" style="zoom:67%;" />

## Structure

```
.
├── .env.example           # Example environment variables file
├── .gitignore             # Git ignore rules file
├── example_history.json   # JSON file containing example history
├── LICENSE                # License file
├── main.py                # Main Python script
├── README.md              # Readme file in English
├── README.zh-CN.md        # Readme file in Simplified Chinese
├── requirements.txt       # File listing required dependencies
├── setup.sh               # Shell script for setup
├── utils.py               # Utility Python script
└── pdf_files              # Directory containing PDF files
    ├── bert.pdf           # PDF file named bert
    └── transformer.pdf    # PDF file named transformer
```

## Principles

1. Read the PDF file and segment it.
2. Generate a feature vector for each text segment using `text-embedding-ada-002`.
3. Generate a feature vector for user input.
4. Calculate the similarity between the user input and the text using `distances_from_embeddings`, and return a list of the most similar texts.
5. Design a prompt and use `GPT3.5-turbo` to generate answers based on the most similar text list.

## Usage

- Download the project

```
git@github.com:Duguce/Mini-ChatPDF.git && cd Mini-ChatPDF
```

- Create a virtual environment

```
./setup.sh
```

- Install the required dependencies

```
pip install -r requirements.txt
```

- Set up environment variables

Obtain a GPT API key from [OpenAI](https://platform.openai.com/account/api-keys) and copy it to the corresponding location in the `.env` file.

- Add documents

Add the PDF documents you want to use in the `./pdf_files/` directory.

- Run the script

```
python3 main.py
```

- Start the conversation

## ToDo

- [x] Support reading multiple PDF documents simultaneously.

- [ ] Support other text encoding vector methods.

- [ ] Add functionality to save text encoding vectors.

- [ ] Implement a graphical user interface (GUI).

- [ ] Optimize the `prompt`.

## License

This project is licensed under the [MIT](https://github.com/Duguce/Mini-ChatPDF/blob/main/LICENSE) License.