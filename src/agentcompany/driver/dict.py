import copy


def flatten_dict(data, parent_key='', sep='.'):
    """
    Flatten a nested dict into a flat dictionary.

    Args:
        data (dict or list): The JSON data to flatten.
        parent_key (str): The base key for recursion (used internally).
        sep (str): The separator used to join keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = {}

    # If the data is a dictionary, iterate through its items.
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            items.update(flatten_dict(value, new_key, sep=sep))
    # If the data is a list, iterate through its elements.
    elif isinstance(data, list):
        for index, value in enumerate(data):
            new_key = f"{parent_key}{sep}{index}" if parent_key else str(index)
            items.update(flatten_dict(value, new_key, sep=sep))
    # Base case: data is not a dict or list, so assign it directly.
    else:
        items[parent_key] = data

    return items

def row_to_nested_dict(row):
    result = {}
    for col in row.index:
        # Convert column name to string explicitly
        str_col = str(col)
        if '.' in str_col:
            main_key, sub_key = str_col.split('.', 1)
            result.setdefault(main_key, {})[sub_key] = row[col]
        else:
            result[str_col] = row[col]
    return result

def unflatten_dict(flat_dict, sep='.'):
    """
    Convert a flattened dictionary to a nested dictionary without numeric bridge keys.
    Handles column names like 'content.text.0' by ignoring trailing numeric segments.
    """
    result = {}
    for compound_key, value in flat_dict.items():
        # Split into key segments
        keys = compound_key.split(sep)
        
        # Remove trailing numeric segments (e.g., '0' in 'content.text.0')
        while keys and keys[-1].isdigit():
            keys.pop()
        if not keys:
            continue  # Skip empty keys after cleanup

        # Build nested dictionary structure
        current = result
        for i, key in enumerate(keys[:-1]):
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    return result

def merge_dicts(original, new_dict):
    """
    Merges two nested dictionaries, preserving all keys from the original dict,
    while adding or updating keys from the new dict. Nested dictionaries are
    merged recursively.
    
    Args:
        original (dict): The base dictionary to merge into
        new_dict (dict): The dictionary with updates/additions
    
    Returns:
        dict: Merged dictionary with combined contents
    """
    merged = copy.deepcopy(original)
    
    for key, new_value in new_dict.items():
        original_value = merged.get(key)
        
        if isinstance(original_value, dict) and isinstance(new_value, dict):
            merged[key] = merge_dicts(original_value, new_value)
        else:
            merged[key] = copy.deepcopy(new_value)
    
    return merged

def dict_rows_to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""

    # Extract headers from keys of first row
    headers = rows[0].keys()

    # Markdown header row
    header_row = "| " + " | ".join(headers) + " |"

    # Markdown separator row
    separator_row = "|" + "|".join(['---'] * len(headers)) + "|"

    # Markdown data rows
    data_rows = []
    for row in rows:
        row_values = [str(row[key]).replace("\n", " ").replace("|", "\\|") for key in headers]
        data_row = "| " + " | ".join(row_values) + " |"
        data_rows.append(data_row)

    # Combine all parts into final Markdown table
    markdown_table = "\n".join([header_row, separator_row] + data_rows)
    return markdown_table
