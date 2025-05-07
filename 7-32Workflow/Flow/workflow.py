# Author: MangguoD

import os
import sys
import json

# 计算当前文件所在绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构造 Sumv2_single 和 DS_inference 模块所在的绝对目录
sumv2_single_dir = os.path.normpath(os.path.join(current_dir, "../../SumDS"))
ds_inference_dir = os.path.normpath(os.path.join(current_dir, "../../DiabetesPDiagLLM/src/train"))

# 将两个模块所在目录添加到 sys.path（注意使用 insert 保证优先级）
if sumv2_single_dir not in sys.path:
    sys.path.insert(0, sumv2_single_dir)
if ds_inference_dir not in sys.path:
    sys.path.insert(0, ds_inference_dir)

# 模型本地路径（常量）
MODEL_7B_PATH = os.path.abspath(os.path.join(current_dir, "../../autodl-tmp/DeepSeek-R1-Distill-Qwen-7B"))
MODEL_32B_PATH = os.path.abspath(os.path.join(current_dir, "../../autodl-tmp/DiabetesPDiagLLM"))

# 引入模块函数
from Sumv2_single import query_llm_single, load_model_and_tokenizer_7b
from DS_inference import load_model_and_tokenizer_32b, inference

def main():
    # 日志文件定义
    log_file_path = os.path.join(current_dir, "workflow_output.log")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("===== 工作流推理日志 =====\n")

    # 加载两个模型
    print("正在加载 DeepSeek-7B 模型...")
    model_7b, tokenizer_7b = load_model_and_tokenizer_7b(MODEL_7B_PATH)
    print("正在加载 DeepSeek-32B 模型...")
    model_32b, tokenizer_32b = load_model_and_tokenizer_32b(MODEL_32B_PATH)

    print("请输入病情描述（直接回车确认，每条处理完继续输入，输入 'quit' 或 'exit' 退出）：")

    counter = 1
    while True:
        user_input = input(">> ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("已退出。")
            break
        if user_input == "":
            continue

        # 结构化预处理（7B）
        print("正在使用SumDS进行预处理...")
        preprocessed = query_llm_single(user_input, model=model_7b, tokenizer=tokenizer_7b)
        print("预处理完成")

        # 深度推理（32B）
        print("正在使用DiabetesPDiagLLM进行推理...")
        final_output = inference(preprocessed, model_32b, tokenizer=tokenizer_32b)

        # 写入日志
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== 第 {counter} 条记录 ===\n")
            log_file.write("[原始输入]\n" + user_input + "\n\n")
            log_file.write("[SumDS结构化输出]\n" + preprocessed + "\n\n")
            log_file.write("[DiabetesPDiagLLM推理结果]\n" + final_output + "\n")
            log_file.write("-" * 30 + "\n")

        print("[推理结果]:\n" + final_output)
        counter += 1

if __name__ == "__main__":
    main()