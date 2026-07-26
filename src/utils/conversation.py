"""
多轮对话管理模块

提供完整的多轮对话数据结构、历史管理和提示词格式化功能。
支持:
- 多角色消息 (system / user / assistant)
- 对话历史持久化 (JSON 序列化)
- 模板化提示词构建
- 上下文窗口自动截断
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


# ============================================================================
# 对话模板定义
# ============================================================================

@dataclass
class ConversationTemplate:
    """
    对话格式化模板 — 将消息列表转换为模型可理解的 prompt 字符串。
    """

    name: str  # 模板名称 (如 "chatml")
    system_start: str = ""  # system 消息前缀
    system_end: str = ""  # system 消息后缀
    user_start: str = ""  # user 消息前缀
    user_end: str = ""  # user 消息后缀
    assistant_start: str = ""  # assistant 消息前缀
    assistant_end: str = ""  # assistant 消息后缀
    separator: str = "\n"  # 消息间分隔符
    suffix: str = ""  # 最终后缀（用于提示模型开始回答）


# 常用模板
CHATML_TEMPLATE = ConversationTemplate(
    name="chatml",
    system_start="<|im_start|>system\n",
    system_end="<|im_end|>\n",
    user_start="<|im_start|>user\n",
    user_end="<|im_end|>\n",
    assistant_start="<|im_start|>assistant\n",
    assistant_end="<|im_end|>\n",
    suffix="<|im_start|>assistant\n",
)

LLAMA3_TEMPLATE = ConversationTemplate(
    name="llama3",
    system_start="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    system_end="<|eot_id|>",
    user_start="<|start_header_id|>user<|end_header_id|>\n\n",
    user_end="<|eot_id|>",
    assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_end="<|eot_id|>",
    suffix="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

SIMPLE_TEMPLATE = ConversationTemplate(
    name="simple",
    system_start="[SYSTEM] ",
    system_end="\n\n",
    user_start="[USER] ",
    user_end="\n",
    assistant_start="[ASSISTANT] ",
    assistant_end="\n",
    suffix="[ASSISTANT] ",
)

# 默认模板
DEFAULT_TEMPLATE = SIMPLE_TEMPLATE


# ============================================================================
# 消息与对话
# ============================================================================

@dataclass
class Message:
    """单条对话消息"""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Unknown role: {self.role} (expected system/user/assistant)")

    def to_dict(self) -> Dict[str, str]:
        """序列化为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Message":
        """从字典反序列化"""
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", ""),
        )


class Conversation:
    """
    多轮对话管理器

    负责维护消息历史、格式化 prompt、以及上下文窗口截断。

    Usage:
        conv = Conversation(system_prompt="你是一个有帮助的助手")
        conv.add_user("今天天气怎么样？")
        conv.add_assistant("今天天气晴朗，适合户外活动。")
        conv.add_user("推荐一个户外活动")
        prompt = conv.build_prompt()  # 返回格式化后的 prompt 字符串
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        template: Optional[ConversationTemplate] = None,
        max_context_length: int = 4096,
    ):
        """
        初始化对话

        Args:
            system_prompt: 系统提示词（设定模型角色/行为）
            template: 对话格式化模板，默认使用 SIMPLE_TEMPLATE
            max_context_length: 上下文窗口最大字符数（超过时自动截断最早的对话）
        """
        self.messages: List[Message] = []
        self.template = template or DEFAULT_TEMPLATE
        self.max_context_length = max_context_length

        # 始终将 system message 作为第一条
        if system_prompt and system_prompt.strip():
            self.messages.append(Message(role="system", content=system_prompt.strip()))

    # ---- 属性 ----

    @property
    def system_prompt(self) -> Optional[str]:
        """获取系统提示词内容"""
        for msg in self.messages:
            if msg.role == "system":
                return msg.content
        return None

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]):
        """设置/更新系统提示词"""
        # 移除旧的 system message
        self.messages = [m for m in self.messages if m.role != "system"]
        if value and value.strip():
            self.messages.insert(0, Message(role="system", content=value.strip()))

    @property
    def history(self) -> List[Message]:
        """返回非 system 的消息历史"""
        return [m for m in self.messages if m.role in ("user", "assistant")]

    @property
    def last_message(self) -> Optional[Message]:
        """返回最后一条消息（系统消息除外）"""
        non_sys = [m for m in self.messages if m.role != "system"]
        return non_sys[-1] if non_sys else None

    # ---- 消息管理 ----

    def add_message(self, role: str, content: str):
        """
        添加一条消息到对话历史

        Args:
            role: 角色 (system/user/assistant)
            content: 消息内容
        """
        if not content or not content.strip():
            return
        self.messages.append(Message(role=role, content=content.strip()))

    def add_user(self, content: str):
        """添加用户消息"""
        self.add_message("user", content)

    def add_assistant(self, content: str):
        """添加助手回复"""
        self.add_message("assistant", content)

    def add_system(self, content: str):
        """添加/更新系统提示词"""
        self.system_prompt = content

    def clear(self, keep_system: bool = True):
        """
        清空对话历史

        Args:
            keep_system: 是否保留系统提示词，默认 True
        """
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []

    # ---- Prompt 构建 ----

    def build_prompt(self, template: Optional[ConversationTemplate] = None) -> str:
        """
        将对话历史格式化为模型输入的 prompt 字符串。

        使用模板规则：
            system  → template.system_start + content + template.system_end
            user     → template.user_start + content + template.user_end
            assistant→ template.assistant_start + content + template.assistant_end

        最后追加 template.suffix，提示模型开始生成 assistant 回复。

        Args:
            template: 可选的自定义模板，默认使用 self.template

        Returns:
            格式化后的 prompt 字符串
        """
        tmpl = template or self.template
        parts = []

        for msg in self.messages:
            if msg.role == "system":
                parts.append(f"{tmpl.system_start}{msg.content}{tmpl.system_end}")
            elif msg.role == "user":
                parts.append(f"{tmpl.user_start}{msg.content}{tmpl.user_end}")
            elif msg.role == "assistant":
                parts.append(f"{tmpl.assistant_start}{msg.content}{tmpl.assistant_end}")

        prompt = tmpl.separator.join(parts)
        if tmpl.suffix:
            prompt += tmpl.separator + tmpl.suffix if prompt else tmpl.suffix

        return prompt

    def build_prompt_truncated(self) -> str:
        """
        构建 prompt 并在超出 max_context_length 时自动截断。

        截断策略：从最早的 user/assistant 对话开始裁剪，
        保留 system message 和最近的 N 轮对话。

        Returns:
            截断后的 prompt 字符串
        """
        prompt = self.build_prompt()
        if len(prompt) <= self.max_context_length:
            return prompt

        # 需要截断：保留 system + 最近的对话
        # 慢慢从前面移除 history 消息直到满足长度
        sys_msgs = [m for m in self.messages if m.role == "system"]
        history_msgs = [m for m in self.messages if m.role != "system"]

        if not history_msgs:
            return prompt  # 只有 system，无法截断

        # 二分查找合适的历史起始位置
        lo, hi = 0, len(history_msgs)
        while lo < hi:
            mid = (lo + hi) // 2
            temp_conv = Conversation(template=self.template, max_context_length=self.max_context_length)
            temp_conv.messages = sys_msgs + history_msgs[mid:]
            temp_prompt = temp_conv.build_prompt()
            if len(temp_prompt) <= self.max_context_length:
                hi = mid
            else:
                lo = mid + 1

        # lo 是第一个使 prompt 满足长度限制的起始索引
        if lo < len(history_msgs):
            self.messages = sys_msgs + history_msgs[lo:]

        return self.build_prompt()

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "template": self.template.name,
            "max_context_length": self.max_context_length,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        """
        序列化为 JSON 字符串。

        Args:
            path: 可选的文件路径，提供则写入文件

        Returns:
            JSON 字符串
        """
        data = self.to_dict()
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_dict(cls, d: Dict[str, Any], template: Optional[ConversationTemplate] = None) -> "Conversation":
        """
        从字典反序列化

        Args:
            d: 序列化字典
            template: 可选的自定义模板
        """
        if template is None:
            template_name = d.get("template", DEFAULT_TEMPLATE.name)
            template = get_template_by_name(template_name)

        conv = cls(
            template=template,
            max_context_length=d.get("max_context_length", 4096),
        )
        conv.messages = [Message.from_dict(m) for m in d.get("messages", [])]
        return conv

    @classmethod
    def from_json(cls, json_str_or_path: str) -> "Conversation":
        """
        从 JSON 字符串或文件路径加载对话

        Args:
            json_str_or_path: JSON 字符串或文件路径
        """
        # 尝试作为文件路径
        import os
        if os.path.exists(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"Conversation(messages={len(self.messages)}, template='{self.template.name}')"


# ============================================================================
# 模板工具函数
# ============================================================================

_template_registry: Dict[str, ConversationTemplate] = {
    "chatml": CHATML_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
    "simple": SIMPLE_TEMPLATE,
}


def get_template_by_name(name: str) -> ConversationTemplate:
    """根据名称获取对话模板"""
    return _template_registry.get(name, DEFAULT_TEMPLATE)


def register_template(template: ConversationTemplate) -> None:
    """注册自定义模板"""
    _template_registry[template.name] = template