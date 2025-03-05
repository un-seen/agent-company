from agentcompany.mcp.base import ModelContextProtocolImpl

class FinalAnswerFunction(ModelContextProtocolImpl):
    
    description = "This is the final answer function."
    name = "final_answer"
    inputs = {"prompt": str}
    output_type = str
     
    def forward(self, answer: str):
        return answer