from agentcompany.mcp.base import ModelContextProtocolImpl


class FinalAnswerFunction(ModelContextProtocolImpl):

    description = "This is the final answer function."
    name = "final_answer"
    inputs = {"prompt": {"type": "string", "description": "The final answer."}}
    output_type = "string"

    def forward(self, answer: str):
        return answer


class AbbaDabba(ModelContextProtocolImpl):

    description = "This is the function that does the abba dabba."
    name = "abba_dabba"
    inputs = {"prompt": {"type": "string", "description": "The prompt to abba dabba."}}
    output_type = "string"

    def forward(self, prompt: str):
        return f"Abba dabba {prompt}"
