from typing import Optional, Set
import re
import os
from pydantic import BaseModel

# Precompile once at import-time
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def quick_word_match(s1: str, s2: str, 
                     *, 
                     case_insensitive: bool = True,
                     word_pattern: Optional[re.Pattern] = None
                    ) -> bool:
    """
    Return True if any “word” in s1 also appears in s2.
    
    - Runs in O(len(s1) + len(s2)) time.
    - Uses a set for O(1) lookups.
    - Short‑circuits on first match for s2.
    
    Params:
      s1, s2            : input strings
      case_insensitive  : if True, lower‑cases before matching
      word_pattern      : custom regex for “words” (default = \\b\\w+\\b)
    """
    pat = word_pattern or _WORD_RE
    
    if case_insensitive:
        s1 = s1.lower()
        s2 = s2.lower()
    
    # build set of words from s1
    words1: Set[str] = {m.group() for m in pat.finditer(s1)}
    if not words1:
        return False
    
    # scan s2, stop at first hit
    for m in pat.finditer(s2):
        if m.group() in words1:
            return True
    return False


class QuestionAnswer(BaseModel):
    question: str
    answer: Optional[str]
    success: bool
    
    
def exa_web_qa(prompt: str) -> str:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.exa.ai",
        api_key=os.environ["EXA_API_KEY"],
    )
    completion = client.chat.completions.create(
        model="exa",
        messages=[
            {"role":"user","content": prompt}    
        ],
    )
    response = completion.choices[0].message.content
    return response
    

def answer_from_data(data: str, question: str) -> QuestionAnswer:
    """
    Generate a JSON array for the user text using the Gemini API.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = QuestionAnswer(question=question, answer=None, success=False)
    # TODO fix the prompt by switching to flow agent here
    prompt = f"""
    You have to answer the question with a plain text value and not formatting based on the below data:
    
    {data} \n\n
    
    Question:
    {question}
    
    You can extrapolate the answer if sufficient information is not provided but you can derive the answer from the data.
    """
    # Generate Answer
    completion = client.chat.completions.create(
        model="o4-mini",
        messages=[{"role":"user","content": prompt}],
    )
    code_action = completion.choices[0].message.content
    print(f"AnswerFromData -> Prompt: {prompt} Code Action: {code_action}")
    # Judge Answer
    judge_prompt = f"""
    Question: {question}
    Answer: {code_action}
    Is any answer given in the answer? Answer with True or False.
    """.strip()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": judge_prompt}],
    )
    judge = completion.choices[0].message.content
    print(f"Answer_from_Data_Judge: {judge}")
    if judge.lower() == "true":
        response.success = True
        response.answer = code_action
    return response