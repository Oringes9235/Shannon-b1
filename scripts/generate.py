#!/usr/bin/env python
"""
流式文本生成脚本 - 支持单次生成和交互式多轮对话
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import readline

import torch

from src.model import ShannonB1, ModelConfig
from src.data import CharTokenizer, BPETokenizer
from src.utils import Conversation, get_template_by_name


def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
    elif "model_config" in checkpoint:
        config = checkpoint["model_config"]
    else:
        state_dict = checkpoint["model_state_dict"]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        d_model = state_dict["token_embedding.weight"].shape[1]
        max_seq_len = state_dict["pos_encoding.pe"].shape[1]
        config = ModelConfig(vocab_size=vocab_size, d_model=d_model, max_seq_len=max_seq_len)

    model = ShannonB1(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_path = model_path.replace(".pt", "_tokenizer.json")
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "char_to_idx" in data:
            tokenizer = CharTokenizer()
            tokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer()
            tokenizer.load(tokenizer_path)
    else:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["sample text"], 1000)

    return model, tokenizer, config


def _extract_assistant_reply(full_text, input_prompt):
    """从完整输出中提取助手的回复部分"""
    if full_text.startswith(input_prompt):
        reply = full_text[len(input_prompt) :].strip()
    else:
        reply = full_text.strip()
    markers = ["[ASSISTANT] ", "[ASSISTANT]", "<|im_start|>assistant\n", "assistant\n"]
    for marker in markers:
        if reply.startswith(marker):
            reply = reply[len(marker) :].strip()
    return reply


def _clean_new_text(text):
    """去除新生成文本中可能的模板标记"""
    markers = [
        "[ASSISTANT]\n", "[ASSISTANT] ", "[ASSISTANT]",
        "<|im_start|>assistant\n", "<|im_start|>assistant",
        "assistant\n", "assistant ",
    ]
    for m in markers:
        if text.startswith(m):
            text = text[len(m) :]
    return text


def single_generate(model, tokenizer, args):
    """单次生成模式"""
    print(f"\n{'='*60}")
    if args.system_prompt:
        print(f"🤖 System Prompt: {args.system_prompt}")
        print(f"{'-'*60}")
    print(f"💬 User Prompt: {args.prompt}")
    print(f"{'='*60}\n")

    # Build full prompt
    if args.system_prompt and args.system_prompt.strip():
        template = get_template_by_name(args.conv_template)
        conv = Conversation(system_prompt=args.system_prompt, template=template)
        conv.add_user(args.prompt)
        full_prompt = conv.build_prompt()
    else:
        full_prompt = args.prompt

    start_tokens = tokenizer.encode(full_prompt)[:50]
    prompt_len = len(start_tokens)
    generated_tokens = list(start_tokens)
    start_time = time.time()

    print("🚀 开始流式生成:\n")
    # Print prompt once, then overwrite with new text each token
    print(f"{args.prompt}", end="", flush=True)

    try:
        for token_id, probability in model.generate_stream(
            start_tokens,
            args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            generated_tokens.append(token_id)
            new_tokens = generated_tokens[prompt_len:]
            new_text = tokenizer.decode(new_tokens)
            new_text = new_text.replace("</w>", " ").replace("  ", " ")
            new_text = _clean_new_text(new_text)

            print(f"\r{args.prompt}{new_text}", end="", flush=True)
            if args.delay > 0:
                time.sleep(args.delay)

        elapsed = time.time() - start_time
        tokens_gen = len(generated_tokens) - prompt_len
        print(f"\n\n{'='*60}")
        print("✅ 生成完成!")
        print(f"📊 统计: {tokens_gen} tokens / {elapsed:.2f}s / {tokens_gen/elapsed:.1f} t/s")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断生成")
    except Exception as e:
        print(f"\n\n❌ 生成出错: {e}")


def interactive_chat(model, tokenizer, args):
    """交互式多轮对话模式"""
    template = get_template_by_name(args.conv_template)

    if args.load_conv:
        try:
            conv = Conversation.from_json(args.load_conv)
            print(f"📂 已加载对话历史: {args.load_conv}")
            print(f"   消息数: {len(conv)}")
            if conv.system_prompt:
                print(f"   系统提示词: {conv.system_prompt}")
            for msg in conv.history[-6:]:
                icon = "🧑" if msg.role == "user" else "🤖"
                preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                print(f"   {icon} [{msg.role}]: {preview}")
        except Exception as e:
            print(f"⚠️  加载对话失败: {e}")
            print("   将创建新对话")
            conv = Conversation(
                system_prompt=args.system_prompt,
                template=template,
                max_context_length=args.max_context,
            )
    else:
        conv = Conversation(
            system_prompt=args.system_prompt,
            template=template,
            max_context_length=args.max_context,
        )

    print(f"\n{'='*60}")
    print("🤖 Shannon-b1 多轮对话模式")
    print(f"{'='*60}")
    print(f"📋 模板: {args.conv_template}")
    if conv.system_prompt:
        print(f"🎯 系统提示词: {conv.system_prompt}")
    print(f"📐 最大上下文: {args.max_context} 字符")
    print(f"\n命令:")
    print(f"  /clear          清空对话历史（保留系统提示词）")
    print(f"  /save [path]    保存对话到文件")
    print(f"  /history        显示对话历史")
    print(f"  /system <text>  修改系统提示词")
    print(f"  /stats          显示对话统计")
    print(f"  /exit           退出对话")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("🧑 你: ").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.split(maxsplit=1)
                command = cmd[0].lower()
                cmd_arg = cmd[1] if len(cmd) > 1 else ""

                if command == "/exit":
                    save_prompt = input("💾 是否保存当前对话? (y/n, 默认n): ").strip().lower()
                    if save_prompt == "y":
                        filename = args.save_path or f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.json"
                        conv.to_json(filename)
                        print(f"✅ 对话已保存到: {filename}")
                    print("👋 再见!")
                    break

                elif command == "/clear":
                    conv.clear(keep_system=True)
                    print("🗑️  对话历史已清空（系统提示词保留）")

                elif command == "/save":
                    filename = cmd_arg or args.save_path or f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    conv.to_json(filename)
                    print(f"✅ 对话已保存到: {filename}")

                elif command == "/history":
                    print(f"\n{'─'*50}")
                    print(f"📜 对话历史 ({len(conv)} 条消息):")
                    print(f"{'─'*50}")
                    for i, msg in enumerate(conv.messages):
                        icon = {"system": "⚙️", "user": "🧑", "assistant": "🤖"}.get(msg.role, "❓")
                        print(f"  [{i}] {icon} {msg.role}: {msg.content[:120]}")
                    print(f"{'─'*50}\n")

                elif command == "/system":
                    if cmd_arg:
                        conv.add_system(cmd_arg)
                        print(f"✅ 系统提示词已更新: {cmd_arg}")
                    else:
                        print(f"当前系统提示词: {conv.system_prompt or '(无)'}")

                elif command == "/stats":
                    total_chars = sum(len(m.content) for m in conv.messages)
                    user_msgs = sum(1 for m in conv.messages if m.role == "user")
                    assistant_msgs = sum(1 for m in conv.messages if m.role == "assistant")
                    print(f"\n📊 对话统计:")
                    print(f"   总消息数: {len(conv)}")
                    print(f"   用户消息: {user_msgs}")
                    print(f"   助手消息: {assistant_msgs}")
                    print(f"   总字符数: {total_chars}")
                    print(f"   模板: {args.conv_template}")
                    print(f"   最大上下文: {args.max_context}\n")

                else:
                    print(f"❓ 未知命令: {command}")
                    print("   可用命令: /clear /save /history /system /stats /exit")

                continue

            # Add user message and generate reply
            conv.add_user(user_input)

            # Print blank line for spacing
            print()

            # Build prompt
            if args.max_context > 0:
                full_prompt = conv.build_prompt_truncated()
            else:
                full_prompt = conv.build_prompt()

            start_tokens = tokenizer.encode(full_prompt)[:50]
            prompt_len = len(start_tokens)
            generated_tokens = list(start_tokens)
            start_time = time.time()
            new_text = ""

            print("🤖 助手: ", end="", flush=True)

            try:
                for token_id, probability in model.generate_stream(
                    start_tokens,
                    args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                ):
                    generated_tokens.append(token_id)
                    new_tokens = generated_tokens[prompt_len:]
                    new_text = tokenizer.decode(new_tokens)
                    new_text = new_text.replace("</w>", " ").replace("  ", " ")
                    new_text = _clean_new_text(new_text)
                    print(f"\r🤖 助手: {new_text}", end="", flush=True)

                # Extract final reply
                current_text = tokenizer.decode(generated_tokens).replace("</w>", " ").replace("  ", " ").strip()
                assistant_reply = _extract_assistant_reply(current_text, full_prompt)
                if not assistant_reply:
                    assistant_reply = new_text

                conv.add_assistant(assistant_reply)

                elapsed = time.time() - start_time
                tokens_gen = len(generated_tokens) - prompt_len
                print(f"\n   ⏱️  {tokens_gen} tokens / {elapsed:.2f}s / {tokens_gen/elapsed:.1f} t/s\n")

            except KeyboardInterrupt:
                print("\n⚠️  生成中断")
                if new_text:
                    partial = _extract_assistant_reply(
                        tokenizer.decode(generated_tokens).replace("</w>", " ").replace("  ", " "),
                        full_prompt,
                    ) or new_text
                    conv.add_assistant(partial + " [中断]")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except EOFError:
            print("\n\n👋 再见!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Shannon-b1 流式文本生成 - 支持单次生成和交互式多轮对话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次生成
  python scripts/generate.py --model-path checkpoints/model.pt --prompt "你好"

  # 交互式多轮对话
  python scripts/generate.py --model-path checkpoints/model.pt -i

  # 多轮对话 + 系统提示词 + ChatML 模板
  python scripts/generate.py --model-path checkpoints/model.pt -i --system-prompt "你是专家" --conv-template chatml

  # 恢复已保存的对话
  python scripts/generate.py --model-path checkpoints/model.pt -i --load my_chat.json
        """,
    )
    parser.add_argument("--model-path", "--checkpoint", type=str, required=True, help="模型文件路径")
    parser.add_argument("--prompt", type=str, default="The ", help="提示词（单次生成模式）")
    parser.add_argument("--system-prompt", type=str, default=None, help="系统提示词（可选）")
    parser.add_argument("--max-tokens", "--max-new-tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top-p", type=float, default=None, help="Top-P采样参数")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚系数")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备")
    parser.add_argument("--delay", type=float, default=0.05, help="每个token之间的延迟（秒）")

    # Multi-turn options
    parser.add_argument("-i", "--interactive", action="store_true", help="启用交互式多轮对话模式（REPL）")
    parser.add_argument("--conv-template", type=str, default="simple",
                        choices=["simple", "chatml", "llama3"], help="对话模板格式 (默认: simple)")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="最大上下文长度（字符数），超出自断。0=不限制 (默认: 4096)")
    parser.add_argument("--load", "--load-conv", dest="load_conv", type=str, default=None,
                        help="从 JSON 文件加载对话历史")
    parser.add_argument("--save", "--save-path", dest="save_path", type=str, default=None,
                        help="保存对话历史的默认路径")

    args = parser.parse_args()

    print("🔄 加载模型...")
    model, tokenizer, config = load_model(args.model_path, args.device)

    if args.interactive:
        interactive_chat(model, tokenizer, args)
    else:
        print(f"✅ 模型加载完成: vocab={config.vocab_size}, d_model={config.d_model}")
        print(f"📝 分词器类型: {'BPE' if hasattr(tokenizer, 'merges') else 'Char'}")
        single_generate(model, tokenizer, args)


if __name__ == "__main__":
    main()