"""
多轮对话管理功能单元测试
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from src.utils import (
    Message,
    Conversation,
    ConversationTemplate,
    SIMPLE_TEMPLATE,
    CHATML_TEMPLATE,
    LLAMA3_TEMPLATE,
    DEFAULT_TEMPLATE,
    get_template_by_name,
    register_template,
)


class TestMessage(unittest.TestCase):
    """测试 Message 数据结构"""

    def test_create_message(self):
        """测试创建消息"""
        msg = Message(role="user", content="你好")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "你好")
        self.assertIsNotNone(msg.timestamp)

    def test_invalid_role(self):
        """测试无效角色抛出异常"""
        with self.assertRaises(ValueError):
            Message(role="invalid", content="test")

    def test_message_serialization(self):
        """测试消息序列化和反序列化"""
        msg = Message(role="assistant", content="好的！")
        d = msg.to_dict()
        self.assertIn("role", d)
        self.assertIn("content", d)
        self.assertIn("timestamp", d)

        restored = Message.from_dict(d)
        self.assertEqual(restored.role, msg.role)
        self.assertEqual(restored.content, msg.content)
        self.assertEqual(restored.timestamp, msg.timestamp)

    def test_empty_content(self):
        """测试空内容消息"""
        msg = Message(role="user", content="")
        self.assertEqual(msg.content, "")

    def test_timestamp_auto_generated(self):
        """测试时间戳自动生成"""
        msg = Message(role="system", content="test")
        self.assertIsInstance(msg.timestamp, str)
        self.assertGreater(len(msg.timestamp), 0)


class TestConversation(unittest.TestCase):
    """测试 Conversation 对话管理"""

    def test_empty_conversation(self):
        """测试空对话"""
        conv = Conversation()
        self.assertEqual(len(conv), 0)
        self.assertIsNone(conv.system_prompt)
        self.assertEqual(conv.history, [])

    def test_with_system_prompt(self):
        """测试带系统提示词的对话"""
        conv = Conversation(system_prompt="你是一个助手")
        self.assertEqual(len(conv), 1)
        self.assertEqual(conv.system_prompt, "你是一个助手")
        self.assertEqual(conv.messages[0].role, "system")

    def test_add_messages(self):
        """测试添加多条消息"""
        conv = Conversation(system_prompt="系统提示词")
        conv.add_user("问题1")
        conv.add_assistant("回答1")
        conv.add_user("问题2")
        conv.add_assistant("回答2")

        self.assertEqual(len(conv), 5)  # system + 2 user + 2 assistant
        self.assertEqual(len(conv.history), 4)  # 不含 system

    def test_system_prompt_property(self):
        """测试 system_prompt 属性 getter/setter"""
        conv = Conversation()
        self.assertIsNone(conv.system_prompt)

        conv.system_prompt = "新系统提示"
        self.assertEqual(conv.system_prompt, "新系统提示")
        self.assertEqual(len(conv), 1)

        # 更新
        conv.system_prompt = "更新的提示"
        self.assertEqual(conv.system_prompt, "更新的提示")
        self.assertEqual(len(conv), 1)  # 仍然只有一条 system

        # 移除
        conv.system_prompt = None
        self.assertIsNone(conv.system_prompt)
        self.assertEqual(len(conv), 0)

    def test_clear_conversation(self):
        """测试清空对话"""
        conv = Conversation(system_prompt="系统")
        conv.add_user("问题")
        conv.add_assistant("回答")

        # 保留 system
        conv.clear(keep_system=True)
        self.assertEqual(len(conv), 1)
        self.assertEqual(conv.messages[0].role, "system")

        # 全部清除
        conv.clear(keep_system=False)
        self.assertEqual(len(conv), 0)

    def test_last_message(self):
        """测试获取最后一条消息"""
        conv = Conversation()
        self.assertIsNone(conv.last_message)

        conv.add_user("问题")
        self.assertEqual(conv.last_message.role, "user")
        self.assertEqual(conv.last_message.content, "问题")

        conv.add_assistant("回答")
        self.assertEqual(conv.last_message.role, "assistant")

    def test_skip_empty_content(self):
        """测试跳过空内容消息"""
        conv = Conversation()
        conv.add_user("")
        conv.add_user("   ")
        self.assertEqual(len(conv), 0)

        conv.add_user("有效内容")
        self.assertEqual(len(conv), 1)


class TestPromptBuilding(unittest.TestCase):
    """测试 Prompt 构建"""

    def setUp(self):
        self.conv = Conversation(system_prompt="你是助手")

    def test_simple_template(self):
        """测试 SIMPLE_TEMPLATE 构建 prompt"""
        self.conv.add_user("你好")
        self.conv.add_assistant("你好！有什么可以帮你？")
        self.conv.add_user("今天天气怎么样")

        prompt = self.conv.build_prompt(template=SIMPLE_TEMPLATE)

        # 应该包含所有消息和 ASSISTANT 后缀
        self.assertIn("[SYSTEM]", prompt)
        self.assertIn("[USER]", prompt)
        self.assertIn("[ASSISTANT]", prompt)
        self.assertIn("你是助手", prompt)
        self.assertIn("你好", prompt)
        self.assertIn("今天天气怎么样", prompt)
        self.assertTrue(prompt.endswith("[ASSISTANT] "))

    def test_chatml_template(self):
        """测试 CHATML_TEMPLATE 构建 prompt"""
        self.conv.add_user("测试")
        prompt = self.conv.build_prompt(template=CHATML_TEMPLATE)

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("assistant\n"))

    def test_llama3_template(self):
        """测试 LLAMA3_TEMPLATE 构建 prompt"""
        self.conv.add_user("测试")
        prompt = self.conv.build_prompt(template=LLAMA3_TEMPLATE)

        self.assertIn("<|start_header_id|>system<|end_header_id|>", prompt)
        self.assertIn("<|eot_id|>", prompt)

    def test_no_system_prompt(self):
        """测试没有系统提示词的 prompt 构建"""
        conv = Conversation()
        conv.add_user("你好")
        prompt = conv.build_prompt(template=SIMPLE_TEMPLATE)

        self.assertNotIn("[SYSTEM]", prompt)
        self.assertIn("[USER] 你好", prompt)

    def test_build_prompt_truncated_no_truncation(self):
        """测试不触发截断的情况"""
        conv = Conversation(system_prompt="短提示词", max_context_length=10000)
        conv.add_user("短问题")
        conv.add_assistant("短回答")

        prompt = conv.build_prompt_truncated()
        self.assertIn("短问题", prompt)
        self.assertIn("短回答", prompt)

    def test_build_prompt_truncated_with_truncation(self):
        """测试触发截断"""
        conv = Conversation(system_prompt="SYS", max_context_length=50)
        conv.add_user("A" * 100)  # 超过 max_context_length
        conv.add_assistant("OK")

        prompt = conv.build_prompt_truncated()
        self.assertLessEqual(len(prompt), 50)
        # system 应该保留
        self.assertIn("SYS", prompt)


class TestConversationSerialization(unittest.TestCase):
    """测试对话序列化/反序列化"""

    def test_to_dict_and_from_dict(self):
        """测试字典序列化/反序列化"""
        conv = Conversation(system_prompt="测试系统", template=SIMPLE_TEMPLATE)
        conv.add_user("用户问题")
        conv.add_assistant("助手回答")

        data = conv.to_dict()
        self.assertIn("messages", data)
        self.assertIn("template", data)
        self.assertEqual(len(data["messages"]), 3)  # system + user + assistant

        restored = Conversation.from_dict(data)
        self.assertEqual(restored.system_prompt, "测试系统")
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored.history[0].content, "用户问题")
        self.assertEqual(restored.history[1].content, "助手回答")

    def test_to_json_and_from_json(self):
        """测试 JSON 序列化/反序列化"""
        conv = Conversation(system_prompt="JSON测试")
        conv.add_user("问题")
        conv.add_assistant("回答")

        json_str = conv.to_json()
        self.assertIsInstance(json_str, str)

        restored = Conversation.from_json(json_str)
        self.assertEqual(restored.system_prompt, "JSON测试")
        self.assertEqual(len(restored), 3)

    def test_to_json_file(self):
        """测试 JSON 文件保存和加载"""
        conv = Conversation(system_prompt="文件测试")
        conv.add_user("问题")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
            conv.to_json(f.name)
            path = f.name

        try:
            restored = Conversation.from_json(path)
            self.assertEqual(restored.system_prompt, "文件测试")
            self.assertEqual(len(restored), 2)
        finally:
            os.unlink(path)


class TestConversationTemplate(unittest.TestCase):
    """测试对话模板"""

    def test_get_template_by_name(self):
        """测试获取模板"""
        tmpl = get_template_by_name("chatml")
        self.assertEqual(tmpl.name, "chatml")

        tmpl = get_template_by_name("simple")
        self.assertEqual(tmpl.name, "simple")

        # 未知名称返回默认
        tmpl = get_template_by_name("unknown")
        self.assertEqual(tmpl.name, DEFAULT_TEMPLATE.name)

    def test_register_custom_template(self):
        """测试注册自定义模板"""
        custom = ConversationTemplate(
            name="my_custom",
            system_start="<<SYS>>",
            system_end="<</SYS>>",
            user_start="<<USR>>",
            user_end="<</USR>>",
            assistant_start="<<BOT>>",
            assistant_end="<</BOT>>",
            suffix="<<BOT>>",
        )
        register_template(custom)

        tmpl = get_template_by_name("my_custom")
        self.assertEqual(tmpl.name, "my_custom")
        self.assertEqual(tmpl.system_start, "<<SYS>>")

    def test_default_template(self):
        """测试默认模板"""
        tmpl = DEFAULT_TEMPLATE
        self.assertEqual(tmpl.name, "simple")
        self.assertEqual(tmpl.system_start, "[SYSTEM] ")
        self.assertEqual(tmpl.user_start, "[USER] ")


def run_tests():
    """运行所有多轮对话测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMessage))
    suite.addTests(loader.loadTestsFromTestCase(TestConversation))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationTemplate))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)