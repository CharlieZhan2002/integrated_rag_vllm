import json
import os
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

class RAGPipeline:
    def __init__(self, json_file="data/rag_dataset.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, json_file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for example in data["examples"]:
            query = example["query"]
            reference_texts = "\n".join(example["reference_contexts"])
            full_content = f"Q: {query}\nA: {reference_texts}"
            documents.append(Document(page_content=full_content))

        # ✅ 使用多语言模型 text2vec-base-multilingual（优先支持英文和中文）
        embedding = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-multilingual",
            model_kwargs={"device": "cpu"}
        )

        self.vectorstore = FAISS.from_documents(documents, embedding)

        self.llm = ChatOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
            model="Qwen2.5-1.5B-Instruct",
            temperature=0.7
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def query(self, question):
        try:
            result = self.qa.invoke(question)
            raw = result.get("result")
            if not isinstance(raw, str):
                raw = str(raw)
            if not raw.strip():
                return "⚠️ 模型未返回有效回答内容。"
            return raw
        except Exception as e:
            return f"❌ RAG 执行失败：{str(e)}"
