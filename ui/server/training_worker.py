"""
训练工作进程 - 使用子进程调用 scripts/train.py 并逐行转发输出
"""

import sys
import os
import subprocess
import threading
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, Callable


class TrainingWorker:
    """后台训练工作进程：通过子进程运行项目的 `scripts/train.py` 并转发输出到回调（通常是 WebSocket）。"""

    def __init__(self, config: Dict[str, Any], callback: Optional[Callable] = None):
        self.config = config
        self.callback = callback
        self.is_running = False
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.status = {
            "is_running": False,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.get("epochs", 30),
            "current_loss": None,
            "best_loss": None,
            "start_time": None
        }

    def run(self):
        """在新线程中启动训练子进程并监听输出。"""
        if self.thread and self.thread.is_alive():
            return

        self.thread = threading.Thread(target=self._run_subprocess, daemon=True)
        self.thread.start()

    def _run_subprocess(self):
        self.is_running = True
        self.status["is_running"] = True
        self.status["start_time"] = datetime.now().isoformat()
        self._send_update("training_started", self.status)

        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            script_path = os.path.join(repo_root, "scripts", "train.py")
            script_path = os.path.normpath(script_path)

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"train.py not found: {script_path}")

            cmd = [sys.executable, script_path]
            # map common config keys to CLI args
            arg_map = {
                "epochs": "--epochs",
                "batch_size": "--batch-size",
                "seq_len": "--seq-len",
                "lr": "--lr",
                "d_model": "--d-model",
                "num_heads": "--num-heads",
                "num_layers": "--num-layers",
                "d_ff": "--d-ff",
                "dropout": "--dropout",
                "tokenizer": "--tokenizer",
                "vocab_size": "--vocab-size",
                "patience": "--patience",
            }

            for k, flag in arg_map.items():
                if k in self.config and self.config[k] is not None:
                    cmd.append(flag)
                    cmd.append(str(self.config[k]))

            # flags
            if self.config.get("no_amp"):
                cmd.append("--no-amp")
            if self.config.get("gradient_checkpointing"):
                cmd.append("--gradient-checkpointing")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # 启动子进程
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=repo_root,
                env=env,
                universal_newlines=True,
                bufsize=1,
            )

            epoch_re = re.compile(r"Epoch\s*(?:[:#])?\s*(\d+)[^\d]+(\d+)")
            # 逐行读取输出并转发
            assert self.proc.stdout is not None
            for raw_line in self.proc.stdout:
                line = raw_line.rstrip("\n")
                self._send_update("training_log", {"line": line})

                # 尝试从日志解析 epoch/progress 信息
                m = epoch_re.search(line)
                if m:
                    try:
                        cur = int(m.group(1))
                        total = int(m.group(2))
                        self.status["current_epoch"] = cur
                        self.status["total_epochs"] = total
                        self.status["progress"] = cur / max(1, total)
                        self._send_update("training_progress", self.status)
                    except Exception:
                        pass

                if not self.is_running:
                    # 用户请求停止：尝试终止子进程
                    break

            # 等待子进程结束
            if self.proc:
                ret = self.proc.wait()
                if ret == 0 and self.is_running:
                    self._send_update("training_completed", {"returncode": ret})
                elif ret != 0:
                    self._send_update("training_error", {"returncode": ret})

        except Exception as e:
            self._send_update("training_error", {"error": str(e)})

        finally:
            # 清理
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception:
                pass

            self.is_running = False
            self.status["is_running"] = False

    def stop(self):
        """停止训练：标记状态并终止子进程。"""
        self.is_running = False
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                # 等待短时间后强杀
                time.sleep(2)
                if self.proc.poll() is None:
                    self.proc.kill()
            except Exception:
                pass
        self._send_update("training_stopped", {"message": "Stopped by user"})

    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return self.status

    def _send_update(self, event: str, data: Dict[str, Any]):
        """通过回调转发事件（含时间戳）。"""
        payload = {
            "type": event,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        if self.callback:
            try:
                self.callback(payload)
            except Exception:
                pass