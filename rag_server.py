# from fastapi import FastAPI, Request, HTTPException
# from pydantic import BaseModel
# from rag_chain import RAGPipeline
# from fastapi.middleware.cors import CORSMiddleware
# import time

# app = FastAPI()
# rag = RAGPipeline()

# # 你定义的 API 密钥
# API_KEY = "123"

# # 添加跨域支持
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 开发阶段放开，生产建议限定源
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 模型列表接口（WebUI 会调用这个）
# @app.get("/rag/models")
# async def get_models():
#     return {
#         "models": [
#             {
#                 "id": "Qwen2.5-1.5B-Instruct/local-rag",
#                 "object": "model",
#                 "created": 0,
#                 "owned_by": "local"
#             }
#         ]
#     }

# # 文档问答接口（你自己的核心逻辑）
# class QueryRequest(BaseModel):
#     question: str

# @app.post("/rag")
# async def ask_rag(req: QueryRequest, request: Request):
#     auth_header = request.headers.get("Authorization")
#     if auth_header != f"Bearer {API_KEY}":
#         raise HTTPException(status_code=401, detail="Unauthorized")
    
#     answer = rag.query(req.question)
#     return {"answer": answer}

# # OpenAI 兼容接口（WebUI 插件会自动调用它）
# @app.post("/rag/chat/completions")
# async def rag_chat_completions(request: Request):
#     auth_header = request.headers.get("Authorization")
#     if auth_header != f"Bearer {API_KEY}":
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     body = await request.json()
#     messages = body.get("messages", [])
    
#     user_input = ""
#     for msg in reversed(messages):
#         if msg.get("role") == "user":
#             user_input = msg.get("content", "")
#             break

#     if not user_input:
#         raise HTTPException(status_code=400, detail="No user message found")

#     answer = rag.query(user_input)

#     # ✅ 返回符合 OpenAI Chat 接口规范的结构
#     return {
#         "id": f"chatcmpl-{int(time.time())}",
#         "object": "chat.completion",
#         "created": int(time.time()),
#         "model": "Qwen2.5-1.5B-Instruct/local-rag",
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": answer
#                 },
#                 "finish_reason": "stop"
#             }
#         ],
#         "usage": {
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#             "total_tokens": 0
#         }
#     }
