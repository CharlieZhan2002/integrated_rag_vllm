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
from datetime import datetime, timedelta

address_cache = {}  # {address_lowercase: {"timestamp": datetime, "txs": List[dict], "status_msg": str}}
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
    now = datetime.utcnow()
    address_lower = address.lower()

    # ✅ 优先检查缓存
    if address_lower in address_cache:
        cached = address_cache[address_lower]
        if now - cached["timestamp"] <= timedelta(minutes=15):
            logger.info(f"⚡ 命中缓存：{address_lower}")
            return cached["txs"], cached["status_msg"]
        else:
            logger.info(f"⏰ 缓存过期，重新查询：{address_lower}")

    # ✅ 没缓存或过期，重新请求
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
                txs = data["result"]
                status_msg = "✅ 成功获取交易数据"
                # ✅ 更新缓存
                address_cache[address_lower] = {
                    "timestamp": now,
                    "txs": txs,
                    "status_msg": status_msg
                }
                return txs, status_msg
            else:
                return [], f"⚠️ 查询失败: {data.get('message', '未知错误')}"
        except Exception as e:
            logger.warning(f"请求错误: {e}")
            retries += 1
            time.sleep(2 ** retries)
    return [], "❌ 多次请求失败"


def process_transactions(txs: List[dict]) -> List[dict]:
    results = []
    for tx in txs:
        try:
            time_fmt = datetime.utcfromtimestamp(int(tx['timeStamp'])).strftime('%Y-%m-%d %H:%M:%S')
            value_eth = round(int(tx['value']) / 1e18, 6)
            gas_price_gwei = round(int(tx['gasPrice']) / 1e9, 2)
            gas_used = int(tx.get('gasUsed', tx['gas']))

            result = {
                "时间": time_fmt,
                "交易哈希": tx['hash'],
                "发送者": tx['from'],
                "接收者": tx.get('to') or 'CONTRACT_CREATION',
                "金额(ETH)": value_eth,
                "input数据": tx.get('input', ''),
                "合约地址": tx.get('contractAddress', ''),
                "实际Gas消耗": gas_used,
                "Gas单价(gwei)": gas_price_gwei,
                "总手续费(ETH)": round((gas_used * int(tx['gasPrice'])) / 1e18, 8),
                "交易状态": "成功" if tx.get('txreceipt_status') == "1" else "失败"
            }
            results.append(result)
        except Exception as e:
            logger.warning(f"处理交易失败: {e}")
            continue
    return results

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

def format_transactions_for_prompt(txs: List[dict]) -> str:
    lines = []
    for tx in txs:
        line = (
            f"时间: {tx['时间']}，交易哈希: {tx['交易哈希']}，"
            f"发送者: {tx['发送者']}，接收者: {tx['接收者']}，"
            f"金额: {tx['金额(ETH)']} ETH，Gas消耗: {tx['实际Gas消耗']}，"
            f"Gas单价: {tx['Gas单价(gwei)']} gwei，总手续费: {tx['总手续费(ETH)']} ETH，"
            f"交易状态: {tx['交易状态']}"
        )
        lines.append(line)
    return "\n".join(lines)

@app.post("/rag")
async def rag_handler(request: RAGRequest):
    prompt = request.prompt.replace("×", "x").strip()
    eth_pattern = r"0x[a-fA-F0-9]{40}"
    addresses = re.findall(eth_pattern, prompt)

    if addresses:
        logger.info("🔍 检测到地址，进入交易处理流程")
        txs, status_msg = get_transactions_from_etherscan(addresses[0])
        tx_context = process_transactions(txs[:10])
        formatted_context = format_transactions_for_prompt(tx_context)  # ✅

        instruction = (
            f"这是用户的问题：{prompt}\n"
            "请你基于下方交易记录，简要回答该问题即可。无需展开输出整张表格，除非问题要求如此。"
        )

        full_prompt = (
            f"你是一个以太坊交易分析助手。\n"
            f"以下是地址 {addresses[0]} 最近的交易记录（最多展示10条）：\n"
            + formatted_context
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

    # ✅ 取最后一条用户输入
    user_message = next((m["content"] for m in request.messages[::-1] if m["role"] == "user"), None)
    if not user_message:
        raise HTTPException(status_code=400, detail="未找到用户消息")

    user_message = user_message.strip()

    # ✅ 检测最后一条用户输入有没有新地址
    eth_pattern = r"0x[a-fA-F0-9]{40}"
    current_addresses = re.findall(eth_pattern, user_message)

    if current_addresses:
        # 🔥 有新地址：说明是想查交易，走交易查询流程
        logger.info("🧠 检测到输入包含地址，走交易查询流程")
        prompt_to_use = user_message
    else:
        # 🔥 没有新地址：就是普通提问，走普通RAG流程
        logger.info("ℹ️ 输入不包含地址，走普通问答流程")
        prompt_to_use = user_message

    # ✅ 只传本轮输入
    rag_response = await rag_handler(RAGRequest(prompt=prompt_to_use))
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
