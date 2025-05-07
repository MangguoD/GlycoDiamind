# Author: MangguoD

import os
import sys
import gradio as gr
import threading
import requests

# 计算当前文件所在绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 Sumv2_single 和 DS_inference 模块所在的绝对目录
sumv2_single_dir = os.path.normpath(os.path.join(current_dir, "../../SumDS"))
ds_inference_dir = os.path.normpath(os.path.join(current_dir, "../../DiabetesPDiagLLM/src/train"))

# 将两个模块所在目录添加到 sys.path
if sumv2_single_dir not in sys.path:
    sys.path.insert(0, sumv2_single_dir)
if ds_inference_dir not in sys.path:
    sys.path.insert(0, ds_inference_dir)

# 模型本地路径
MODEL_7B_PATH = os.path.abspath(os.path.join(current_dir, "../../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"))
MODEL_32B_PATH = os.path.abspath(os.path.join(current_dir, "../../autodl-tmp/DiabetesPDiagLLM"))

# 引入模型函数
from Sumv2_single import query_llm_single, load_model_and_tokenizer_7b
from DS_inference import load_model_and_tokenizer_32b, inference

# 加载模型
print("[初始化] 正在加载 SumDS 模型...")
model_7b, tokenizer_7b = load_model_and_tokenizer_7b(MODEL_7B_PATH)
print("[初始化] 正在加载 DiabetesPDiagLLM 模型...")
model_32b, tokenizer_32b = load_model_and_tokenizer_32b(MODEL_32B_PATH)

# 聊天函数
def chat_with_models(user_input, history):
    history = history or []
    history.append((user_input, "正在思考中..."))
    
    # 结构化预处理
    preprocessed = query_llm_single(user_input, model=model_7b, tokenizer=tokenizer_7b)
    
    # 深度推理
    final_output = inference(preprocessed, model_32b, tokenizer=tokenizer_32b)

    # 返回更新后的聊天记录
    history[-1] = (user_input, final_output)
    return history, history

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 医疗辅助对话系统 \n多模态模型结合，提供结构化分析与专业诊疗建议。")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="请输入患者病情描述，例如：\n患者，女，65岁，血糖波动...", label="输入病情描述后回车发送")
    clear = gr.Button("清除历史记录")

    state = gr.State([])

    msg.submit(chat_with_models, [msg, state], [chatbot, state])
    clear.click(lambda: [], None, chatbot)

# 启动服务（开放端口）
print("正在启动 Gradio 前端服务...")
demo.launch(server_name="0.0.0.0", server_port=6006 ,share=True)
