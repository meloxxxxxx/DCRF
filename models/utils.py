import re
def extract_proofs(proofs_text):
    """
    Lightweight extractor that supports shorthand notation.
    Handles: premise, p, subconclusion, sub, sc.
    Accepts ranges such as premise 1, 2, 3 or p 1-3.
    """
    # print(f"extracting proofs:\n{proofs_text}")
    proofs_text = proofs_text.lower().replace(' and ', ', ')
    if not proofs_text:
        return {'subconclusions': [], 'premises': []}
    
    cleaned_text = re.sub(r'\s+', ' ', proofs_text.strip())
    
    # Extract premises and their aliases (including ranges).
    premise_patterns = [
        r'\[premise\s*(\d+(?:\s*[,-]\s*\d+)*)\]',  # [premise 1, 2, 3] or [premises 1-3]
        r'premise\s*(\d+(?:\s*[,-]\s*\d+)*)',  # premise 1, 2, 3 or premise 1-3
        r'premises\s*(\d+(?:\s*[,-]\s*\d+)*)',  # premises 1, 2, 3 or premises 1-3
        r'premise\s*(\d+(?:\s*[,-]\s*\d+)*)',  # premise 1, 2, 3 or premise 1-3
        r'p\s*(\d+(?:\s*[,-]\s*\d+)*)'         # p 1, 2, 3 or p 1-3
    ]
    
    premises = []
    for pattern in premise_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            premises.extend(expand_number_range(match))
    
    # Extract subconclusions and their aliases (including ranges).
    subconclusion_patterns = [
        r'\[subconclusion\s*(\d+(?:\s*[,-]\s*\d+)*)\]',  # [subconclusion 1, 2, 3]
        r'subconclusions\s*(\d+(?:\s*[,-]\s*\d+)*)',  # subconclusions 1, 2, 3
        r'subconclusion\s*(\d+(?:\s*[,-]\s*\d+)*)',  # subconclusion 1, 2, 3
        r'sub\s*(\d+(?:\s*[,-]\s*\d+)*)',            # sub 1, 2, 3
        r'sc\s*(\d+(?:\s*[,-]\s*\d+)*)'              # sc 1, 2, 3
    ]
    
    subconclusions = []
    for pattern in subconclusion_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            subconclusions.extend(expand_number_range(match))
    
    # Deduplicate and sort identifiers.
    premises = sorted(list(set(premises)), key=lambda x: int(x))
    subconclusions = sorted(list(set(subconclusions)), key=lambda x: int(x))
    
    return {
        'subconclusions': subconclusions,
        'premises': premises
    }

def expand_number_range(range_str):
    """
    Expand numeric ranges into individual tokens.
    Examples:
        "1, 2, 3" -> ['1', '2', '3']
        "1-3" -> ['1', '2', '3']
        "1, 3-5" -> ['1', '3', '4', '5']
    """
    numbers = []
    parts = re.split(r'\s*,\s*', range_str)
    
    for part in parts:
        part = part.strip()
        # Check for an explicit range such as "1-3".
        range_match = re.match(r'(\d+)\s*-\s*(\d+)', part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            numbers.extend(str(i) for i in range(start, end + 1))
        elif part.isdigit():
            numbers.append(part)
    
    return numbers

def extract_answer(answer_text):
    #print(f"extracting answer:\n{answer_text}")
    answer_text = answer_text.lower()
    if any(word in answer_text for word in ['true', 'correct', 'valid', 'right']):
        return 'TRUE'
    elif any(word in answer_text for word in ['false', 'incorrect', 'invalid', 'wrong']):
        return 'FALSE'
    elif any(word in answer_text for word in ['uncertain', 'unknown', 'ambiguous', 'inconclusive']):
        return 'UNCERTAIN'

def extract_logic_components(text):
    """
    Flexibly extract subconclusions, answers, and proofs from multiple formats.
    
    Args:
        text: Raw string that contains the reasoning result.
    
    Returns:
        dict: Parsed structure with extracted components.
    """
    result = {}
    
    # Flexible regex to match "[SUBCONCLUSION X] ... [PROOFS] ...".
    subconclusion_pattern = r'\[SUBCONCLUSION\s+(\d+)\](.*?)\[PROOFS\](.*?)(?=\[SUBCONCLUSION|\n\[ANSWER|\n\[|$)'
    
    # Allow '.' to match newlines so multi-line sections work.
    subconclusion_matches = re.findall(subconclusion_pattern, text, re.DOTALL)
    
    for match in subconclusion_matches:
        sub_num = match[0].strip()
        sub_text = match[1].strip()
        proof_text = match[2].strip()
        
        # 清理文本中的多余空白字符
        # sub_text = re.sub(r'\s+', ' ', sub_text)
        # proof_text = re.sub(r'\s+', ' ', proof_text)
        
        result[sub_num] = {
            'content': sub_text,
            'proofs': extract_proofs(proof_text),
            'verified': False
        }
    
    # Extract the answer along with its proofs.
    answer_pattern = r'\[ANSWER\](.*?)\[PROOFS\](.*?)(?=\[|$)'
    answer_match = re.findall(answer_pattern, text, re.DOTALL)
    
    if answer_match:
        answer_text = answer_match[0][0].strip()
        answer_proofs = answer_match[0][1].strip()
        
        # 清理文本
        answer_text = re.sub(r'\s+', ' ', answer_text)
        answer_proofs = re.sub(r'\s+', ' ', answer_proofs)
        
        result['answer'] = {'content': extract_answer(answer_text),
                            'proofs': extract_proofs(answer_proofs),
                            'verified': False
        }
    
    return result

