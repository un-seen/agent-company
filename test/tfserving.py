from agentcompany.driver.models import OpenAIServerModel
from agentcompany.lib.agents.tfserving import TfServingAgent

if __name__ == "__main__":
    
    model = OpenAIServerModel("gpt-4o-mini")
    tfserving_config = [{
        "model_name": "findings",
        "predict_endpoint": "http://localhost:8501/v1/models/findings:predict",
        "description": f"""
        The model returns a likelihood of the presence of a finding in an input image.
        The value ranges from 0.0 (no finding) to 1.0 (high likelihood of a finding).
        When a mammogram reveals a finding, it indicates that there is a mass, calcification, architectural distortion, or asymmetry.
        This finding could range from benign (non-cancerous) to potentially malignant (cancerous), and it requires further evaluation to determine its significance. 
        In contrast, a normal mammogram means no findings were detected, and the breast tissue appears symmetrical and healthy.
        """,
    }]
    agent = TfServingAgent(
        model=model,
        tfserving_config=tfserving_config,
    )
    data = agent.run("Predict findings on image https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqpkZXoZm5bNW21bVXxWBKV57hrK8nxSHsyg&s")
    print(f"Result")
    print(str(data)[:100] + "...")