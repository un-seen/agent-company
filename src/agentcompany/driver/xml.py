import xml.etree.ElementTree as ET

def plan_xml_to_dict(xml_str):
    import xml.etree.ElementTree as ET
    import re

    # Remove enclosing triple backticks, language identifier, and optional whitespace/newlines
    xml_str = re.sub(r'^```.*?\n', '', xml_str.strip(), flags=re.DOTALL)
    xml_str = re.sub(r'```$', '', xml_str.strip()).strip()

    def process_element(element):
        tag = element.tag
        children = list(element)
        text = (element.text or '').strip()

        # Handle elements that are list containers (ending with '_list')
        if tag.endswith('_list'):
            return [process_element(child) for child in children]
        else:
            if children:
                child_dict = {}
                for child in children:
                    child_data = process_element(child)
                    child_tag = child.tag
                    if child_tag in child_dict:
                        existing = child_dict[child_tag]
                        if isinstance(existing, list):
                            existing.append(child_data)
                        else:
                            child_dict[child_tag] = [existing, child_data]
                    else:
                        child_dict[child_tag] = child_data
                return child_dict
            else:
                return text if text else ''

    root = ET.fromstring(xml_str)
    return {root.tag: process_element(root)}


def plan_dict_to_markdown_without_status(input_dict):
    # Validate input dict structure
    try:
        steps = input_dict['plan']['step']
        assert isinstance(steps, list)
        for step in steps:
            assert 'task' in step
            assert 'function_call_list' in step
            assert isinstance(step['function_call_list'], list)
            for call in step['function_call_list']:
                assert 'function_name' in call
                assert 'fuction_argument_list' in call
                assert isinstance(call['fuction_argument_list'], list)
                for arg in call['fuction_argument_list']:
                    assert 'name' in arg and 'value' in arg
    except (KeyError, AssertionError, TypeError):
        return None

    # Collect unique function names
    func_names = set()
    for step in steps:
        for call in step['function_call_list']:
            func_names.add(call['function_name'])

    func_names = sorted(func_names)

    # Markdown table header
    headers = ['Task'] + func_names + ['Status']
    markdown = '| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'

    # Build rows
    for step in steps:
        row = [step['task']]
        for func in func_names:
            calls = [c for c in step['function_call_list'] if c['function_name'] == func]
            if calls:
                call_details = []
                for call in calls:
                    args = '<br>'.join([f"- {arg['name']}: {arg['value']}" for arg in call['fuction_argument_list']])
                    call_details.append(f"**Arguments:**<br>{args}")
                cell = '<br><br>'.join(call_details)
            else:
                cell = '—'
            row.append(cell)
        row.append('step')
        markdown += '| ' + ' | '.join(row) + ' |\n'

    return markdown

def plan_dict_to_markdown_with_status(input_dict):
    steps = input_dict.get('plan', {}).get('step', [])

    if not steps:
        return "No steps available."

    # Collect unique function names
    func_names = sorted({call['function_name'] for step in steps for call in step.get('function_call_list', [])})

    # Table header
    headers = ['Task'] + func_names + ['Status']
    markdown = '| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'

    # Populate rows
    for step in steps:
        row = [step.get('task', '')]

        for func in func_names:
            calls = [c for c in step.get('function_call_list', []) if c['function_name'] == func]
            if calls:
                details = []
                for call in calls:
                    args = '<br>'.join(f"- {arg['name']}: {arg['value']}" for arg in call.get('fuction_argument_list', []))
                    details.append(f"**Arguments:**<br>{args}")
                cell = '<br><br>'.join(details)
            else:
                cell = '—'
            row.append(cell)

        row.append(step.get('status', ''))
        markdown += '| ' + ' | '.join(row) + ' |\n'

    return markdown

def step_dict_to_xml(input_dict, root_tag='step'):
    def build_element(parent, key, value):
        if isinstance(value, list):
            list_elem = ET.SubElement(parent, key)
            for item in value:
                item_tag = key[:-5] if key.endswith('_list') else 'item'
                build_element(list_elem, item_tag, item)
        elif isinstance(value, dict):
            elem = ET.SubElement(parent, key)
            for k, v in value.items():
                if k != 'status':  # explicitly exclude 'status' key
                    build_element(elem, k, v)
        else:
            elem = ET.SubElement(parent, key)
            elem.text = str(value)

    root = ET.Element(root_tag)
    for k, v in input_dict.items():
        if k != 'status':  # explicitly exclude 'status' key
            build_element(root, k, v)

    return ET.tostring(root, encoding='unicode')


def step_xml_to_dict(xml_str):
    def element_to_dict(element):
        children = list(element)
        if not children:
            return element.text.strip() if element.text else ''

        result = {}
        for child in children:
            child_result = element_to_dict(child)

            if child.tag.endswith('_list'):
                result[child.tag] = child_result if isinstance(child_result, list) else [child_result]
            else:
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_result)
                    else:
                        result[child.tag] = [result[child.tag], child_result]
                else:
                    result[child.tag] = child_result

        # Special handling for '_list' container
        if element.tag.endswith('_list'):
            return [result[child.tag] for child in children]

        return result

    root = ET.fromstring(xml_str)
    return {root.tag: element_to_dict(root)}

def plan_dict_to_xml(input_dict):
    def build_element(parent, key, value):
        if isinstance(value, list):
            list_elem = ET.SubElement(parent, key)
            for item in value:
                singular_tag = key[:-5] if key.endswith('_list') else 'item'
                build_element(list_elem, singular_tag, item)
        elif isinstance(value, dict):
            elem = ET.SubElement(parent, key)
            for k, v in value.items():
                if k != 'status':  # explicitly exclude 'status' key
                    build_element(elem, k, v)
        else:
            elem = ET.SubElement(parent, key)
            elem.text = str(value)

    root_tag = 'plan'
    root = ET.Element(root_tag)

    for k, v in input_dict[root_tag].items():
        build_element(root, k, v)

    return ET.tostring(root, encoding='unicode')