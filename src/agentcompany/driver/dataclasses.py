from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant", "system"]  # Restricts role to specific string literals
    content: Union[str, Dict[str, Any]]
    notebook: Optional[str] = None                 # Optional field with default None
    timestamp: Optional[datetime] = None          # Optional field with default None
    variables: Optional[List[Dict[str, str]]] = None  # Optional list of key-value pairs