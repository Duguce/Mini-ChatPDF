[English](https://github.com/Duguce/Mini-ChatPDF/blob/main/README.md) | 中文简体

# Mini-ChatPDF

本项目是基于 `GPT3.5-turbo`实现，可以根据用户传入的PDF文本文件，回答用户的问题。

以下是一个简单的示例：

<img src="https://zhgyqc.oss-cn-hangzhou.aliyuncs.com/snipaste_20230612_164624.jpg" alt="效果展示" style="zoom:67%;" />

## 项目结构

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

## 基本原理

1. 读取PDF文件，并进行分割；
2. 对于每段文本，基于`text-embedding-ada-002`生成特征向量；
3. 对于用户输入，生成特征向量；
4. 基于`distances_from_embeddings`计算用户输入和文本的相似度，返回最相似的文本列表；
5. 设计`prompt`，并基于 `GPT3.5-turbo`，使其基于最相似的文本列表进行回答。

## 使用方法

- 下载项目

```
git@github.com:Duguce/Mini-ChatPDF.git && cd Mini-ChatPDF
```

- 创建虚拟环境

```
setup.sh 
```

- 安装所需依赖包

```
pip install -r requirements.txt
```

- 设置环境变量

从 [OpenAI](https://platform.openai.com/account/api-keys) 获取一个 GPT API key，并将其复制到`.env`文件的对应位置。

- 添加文档

在`./pdf_files/`路径下添加你所需使用的PDF文档。

- 运行脚本

```
python3 main.py
```

- 开始对话

## 待做功能

- [x] 支持同时读取多个PDF文档
- [ ] 支持其他的文本编码向量方式
- [x] 添加保存文本编码向量的功能
- [ ] 实现可视化界面
- [ ] 优化`prompt`

## 开源许可

本项目采用 [MIT](https://github.com/Duguce/Mini-ChatPDF/blob/main/LICENSE) 开源协议进行许可。