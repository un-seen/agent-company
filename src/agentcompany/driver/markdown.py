import json


def json_to_markdown(json_string: str) -> str:
  try:
    json_contents = json.loads(json_string)
  except json.JSONDecodeError:
    return json_string
  md = ""
  if isinstance(json_contents, list):  
    
    for index, _dict in enumerate(json_contents):
      md += f'| Num: {index} |  |\n| - | - |\n'
      for key in _dict:
        md += f'| {key} | {json_contents[index][key]} |\n'
      md += '---\n'
  else:
    for key in json_contents:
      md += f'| {key} | {json_contents[key]} |\n'
  return md

from typing import List, Dict, Any

def list_of_dict_to_markdown_table(data: List[Dict[str, Any]]) -> str:
    if not data:
        return ""

    headers = data[0].keys()
    table = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(['---'] * len(headers)) + " |"]

    for row in data:
        table.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    return "\n".join(table)
  