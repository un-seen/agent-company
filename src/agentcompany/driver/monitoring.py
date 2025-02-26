import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from redis import Redis
import os

class Monitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if getattr(self.tracked_model, "last_input_token_count", "Not found") != "Not found":
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """Update the metrics of the monitor.

        Args:
            step_log ([`MemoryStep`]): Step log to update the monitor with.
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations) - 1}: Duration {step_duration:.2f} seconds"

        if getattr(self.tracked_model, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(console_outputs, level=1)


class LogLevel(IntEnum):
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"


class AgentLogger:
    def __init__(self, name: str, company_name: str, level: LogLevel = LogLevel.INFO, use_redis: bool = False):
        self.name = name
        self.company_name = company_name
        self.level = level
        self.console = Console()
        if use_redis:
            self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
            
    def log(self, *args, level: str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)
            if hasattr(self, "redis_client"):
                data_dict = kwargs.copy()
                data_dict["message"] = " ".join([str(arg) for arg in args])
                data_dict["level"] = level.name
                data_dict["name"] = self.name
                data_dict["company_name"] = self.company_name
                self.redis_client.publish(self.company_name, json.dumps(data_dict))


__all__ = ["AgentLogger", "Monitor"]