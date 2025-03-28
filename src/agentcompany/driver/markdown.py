import json


def json_to_markdown(json_path):
  json_contents = json.loads(open(json_path).read())
  
  if isinstance(json_contents, list):
    if "/" in json_path:
      md = '# ' + json_path[json_path.rindex("/")+1:json_path.rindex('.')] + '\n'
    else:
      md = '# ' + json_path[:json_path.rindex('.')] + '\n'
    
    for index, _dict in enumerate(json_contents):
      md += f'| Num: {index} |  |\n| - | - |\n'
      for key in _dict:
        md += f'| {key} | {json_contents[index][key]} |\n'
      md += '---\n'
  else:
    for key in json_contents:
      md += f'| {key} | {json_contents[key]} |\n'
  return md