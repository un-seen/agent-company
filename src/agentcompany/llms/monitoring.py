import json
from enum import IntEnum
from datetime import datetime, timezone
from redis import Redis
import os
from rich.console import Console
from rich.markdown import Markdown


class LogLevel(IntEnum):
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"


def print_formatted_content(data: dict) -> None:
    """
    Prints formatted markdown content with colored output using Rich.
    
    Args:
        data (dict): Dictionary containing 'title' and 'text' keys
    """
    console = Console()
    
    # Create markdown content
    if "title" not in data:
        data["title"] = ""
    if "text" not in data:
        data["text"] = ""
    markdown_content = f"# {data['title']}\n\n{data['text']}"
    
    # Print formatted markdown
    console.print(Markdown(markdown_content))


class AgentLogger:
    def __init__(self, name: str, interface_id: str, level: LogLevel = LogLevel.INFO, use_redis: bool = False):
        self.name = name
        self.interface_id = interface_id
        self.level = level
        self.console = Console()
        if use_redis:
            self.redis_client = Redis.from_url(os.environ["REDIS_URL"])
            self.redis_client.delete(f"{self.interface_id}/{self.name}/log")
    
    def set_level(self, level: LogLevel):
        self.level = level

    def log(self, level: str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            # Print to redis
            if hasattr(self, "redis_client"):
                data_dict = {}
                data_dict["role"] = self.name
                data_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
                if not "content" in data_dict or not isinstance(data_dict["content"], dict):
                    data_dict["content"] = {}
                data_dict["content"].update(kwargs)
                self.redis_client.rpush(f"{self.interface_id}/{self.name}/log", json.dumps(data_dict))
            # Print to console
            print_formatted_content(kwargs)


__all__ = ["AgentLogger", "Monitor"]