from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from typing import List
import re
import logging
import time
from collections import deque
from rag_chain import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integrated-server")

# Load env
load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
if not ETHERSCAN_API_KEY:
    logger.warning("⚠️ ETHERSCAN_API_KEY 未设置")

# Address alias
address_map = {}
address_counter = 0

def get_address_alias(address: str) -> str:
    global address_counter
    if address in address_map:
        return address_map[address]
    alias = f"ADDR_{address_counter}"
    address_map[address] = alias
    address_counter += 1
    logger.info(f"地址映射：{address} → {alias}")
    return alias

# Etherscan rate limiter
class EtherscanRateLimiter:
    def __init__(self, calls_per_second=5, max_retries=3):
        self.calls_per_second = calls_per_second
        self.request_timestamps = deque(maxlen=calls_per_second)
        self.max_retries = max_retries

    def wait_if_needed(self):
        now = time.time()
        if len(self.request_timestamps) < self.calls_per_second:
            self.request_timestamps.append(now)
            return
        oldest = self.request_timestamps[0]
        if now - oldest < 1.0:
            time.sleep(1.0 - (now - oldest))
        self.request_timestamps.append(time.time())

etherscan_rate_limiter = EtherscanRateLimiter()

def get_transactions_from_etherscan(address: str) -> (List[dict], str):
    logger.info(f"🔍 查询地址交易：{address}")
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY
    }
    retries = 0
    while retries <= etherscan_rate_limiter.max_retries:
        try:
            etherscan_rate_limiter.wait_if_needed()
            res = requests.get(url, params=params)
            data = res.json()
            if data.get("status") == "1":
                return data["result"], "✅ 成功获取交易数据"
            else:
                return [], f"⚠️ 查询失败: {data.get('message', '未知错误')}"
        except Exception as e:
            logger.warning(f"请求错误: {e}")
            retries += 1
            time.sleep(2 ** retries)
    return [], "❌ 多次请求失败"

def process_transactions(txs: List[dict]) -> List[str]:
    return [
        f"交易哈希: {tx['hash']}，来自: {get_address_alias(tx['from'])}，去往: {get_address_alias(tx.get('to') or 'CONTRACT_CREATION')}，金额: {tx['value']}，gas: {tx['gas']}"
        for tx in txs
    ]

# App & CORS
app = FastAPI(title="RAG + ETH Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# RAG init
rag = RAGPipeline()

# Models
class RAGRequest(BaseModel):
    prompt: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list

@app.post("/rag")
async def rag_handler(request: RAGRequest):
    prompt = request.prompt.replace("×", "x").strip()
    eth_pattern = r"0x[a-fA-F0-9]{40}"
    addresses = re.findall(eth_pattern, prompt)

    if addresses:
        logger.info("🔍 检测到地址，进入交易处理流程")
        txs, status_msg = get_transactions_from_etherscan(addresses[0])
        tx_context = process_transactions(txs[:10])

        instruction = (
            f"这是用户的问题：{prompt}\n"
            "请你基于下方交易记录，简要回答该问题即可。无需展开输出整张表格，除非问题要求如此。"
        )

        full_prompt = (
            f"你是一个以太坊交易分析助手。\n"
            f"以下是地址 {addresses[0]} 最近的交易记录（最多展示10条）：\n"
            + "\n".join(tx_context)
            + f"\n\nEtherscan 查询状态：{status_msg}\n\n{instruction}"
        )

        logger.info("📤 Prompt to LLM:\n" + full_prompt)
        try:
            answer = rag.query(full_prompt)
            if not isinstance(answer, str):
                answer = str(answer)
            if not answer.strip():
                answer = "⚠️ 模型未返回有效回答内容。"
        except Exception as e:
            answer = f"❌ 模型执行失败：{str(e)}"

        return {
            "response": answer,
            "etherscan_status": status_msg,
            "tx_count": len(txs),
            "llm_prompt": full_prompt
        }
    else:
        logger.info("ℹ️ 未检测到地址，走普通问答流程")
        try:
            answer = rag.query(prompt)
            if not isinstance(answer, str):
                answer = str(answer)
            if not answer.strip():
                answer = "⚠️ 模型未返回有效回答内容。"
        except Exception as e:
            answer = f"❌ 模型执行失败：{str(e)}"

        return {"response": answer, "llm_prompt": prompt}

@app.post("/rag/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"🧾 LLM messages: {request.messages}")
    user_message = next((m["content"] for m in request.messages[::-1] if m["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="未找到用户消息")

    user_message = user_message.strip()

    # 🚫 只使用当前用户输入进行判断
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", user_message):
        logger.info("🧠 检测到用户只输入了地址，自动附加默认问题")
        user_message += " 的所有交易数据是什么？"

    # 只传本轮用户输入到 rag_handler
    rag_response = await rag_handler(RAGRequest(prompt=user_message))
    answer = rag_response.get("response", "⚠️ 无法生成回答")

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "Qwen2.5-1.5B-Instruct/local-rag",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer or "⚠️ 无法生成回答"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@app.get("/address-map")
async def get_address_map():
    return address_map

@app.delete("/address-map")
async def clear_address_map():
    global address_map, address_counter
    address_map.clear()
    address_counter = 0
    return {"message": "地址映射已清空"}

@app.get("/")
async def root():
    return {"message": "欢迎使用 RAG + 实时交易查询 API"}