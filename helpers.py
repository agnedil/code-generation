
import re
import os
import gzip
import json
import torch
from typing import Dict, Iterable, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# max sequence length
MAX_LEN = 2048


def get_tokenizer(model_name: str, ACCESS_TOKEN: str | None = None) -> Any:

    if ACCESS_TOKEN:
        tokenizer = AutoTokenizer.from_pretrained( model_name,
                                                   trust_remote_code=True,
                                                   token=ACCESS_TOKEN, )
    else:
        tokenizer = AutoTokenizer.from_pretrained( model_name,
                                                   trust_remote_code=True, )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.pad_token = '<pad>'
    return tokenizer


def get_model(model_name: str, torch_dtype = torch.float32, ACCESS_TOKEN: str|None = None) -> Any:
    ''' Return model depending on the model name.
        Note: * float16 - more precise but has narrow range,
              * bfloat16 less precision, but dynamic range closer to float32 -> useful for DL
              * float32 doesn't have a bfloat32 equivalent as it is already sufficient.
              * Solar 'auto' gets me bfloat16, when I ask for float32 - CUDA out of memory!
              * model.eval() sets model to evaluation mode, which disables dropout,
                sets batch normalization layers to use their running statistics,
                and generally ensures more deterministic behavior

              TODO: test speed vs. accuracy for bloat16 vs. float32 (two separate runs)
    '''
    if ACCESS_TOKEN:
        model = AutoModelForCausalLM.from_pretrained( model_name,
                                                      device_map='cuda',
                                                      trust_remote_code=True,
                                                      torch_dtype=torch_dtype,        # bfloat16, float32, 'auto'
                                                      token = ACCESS_TOKEN, )
    else:
        model = AutoModelForCausalLM.from_pretrained( model_name,
                                                      device_map='cuda',
                                                      trust_remote_code=True,                                                      
                                                      torch_dtype=torch_dtype,        # bfloat16, float32, 'auto'
                                                     )
    model.eval()
    return model


def generate_response(prompt, tokenizer, model, temperature=1.0, top_p=1.0, do_sample=False):
    ''' Tokenize input and make one inference call '''
    messages=[
      { 'role': 'user', 'content': prompt }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=MAX_LEN,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        #top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    res = tokenizer.decode( outputs[0][len(inputs[0]):], skip_special_tokens=True, )
    return res


def clean_code(code: str, signature: str=None) -> str:
    '''
    Clean code by:
        * Extracting only code between "```python" and "```" code fences.
          Other opening code fences include: "```python3", "```Python", or "```Python3".
        * Remove any lines w/test cases starting with 'assert'
        * Remove any lines w/test cases starting with "print(function_name(".
        * Add func header if missing
        * Add import statements if missing, including 'from typing import *'
        * Remove the 'main' body
        * Remove some irrelevant lines
    '''
    # edge cases
    if not isinstance(code, str):
        print('WARNING. Not a string passed for evaluation of generated code - returning an empty string')
        return ''
    if not code:
        return code
    
    # implement steps to clean code
    code    = get_code_in_fences(code)                     # parse code out of code fences
    code    = split_code_on_main(code)                     # keep functions, remove main body
    code    = remove_assert_statements(code)               # remove lines starting with 'assert'    
    code    = remove_print_function_statements(code)       # remove lines starting with "print(function_name("
    code    = remove_irrelevant_lines(code)                # remove specific irrelevant lines
    if signature:
        is_func = is_function(code)
        if not is_func:                                    # add func signature if missing
            code = join_properly(signature, code)
        else:
            code = merge_imports_with_code(signature, code)    # add all original import statements
    code = add_typing_import(code)                         # add from typing import * (if missing)
    return code.strip()


def clean_code_light(code: str, signature: str=None) -> str:
    '''
    Clean code by:
        * Extracting only code between "```python" and "```" code fences.
          Other opening code fences include: "```python3", "```Python", or "```Python3".
        * Add func header if missing
        * Add import statements if missing, including 'from typing import *'
        * Remove the 'main' body
        * Remove some irrelevant lines

    Compared with clean_code() above, this function doesn't do this:
        * Remove any lines w/test cases starting with 'assert'
        * Remove any lines w/test cases starting with "print(function_name(".
        
    Reason: supposedly, these tests don't break the code execution while they may break it after
            an attempt to remove them, because they may span more than one line - TBD.
    '''
    # edge cases
    if not isinstance(code, str):
        print('WARNING. Not a string passed for evaluation of generated code - returning an empty string')
        return ''
    if not code:
        return code
    
    # implement steps to clean code
    code    = get_code_in_fences(code)                     # parse code out of code fences
    code    = split_code_on_main(code)                     # keep functions, remove main body
    #code    = remove_assert_statements(code)              # remove lines starting with 'assert'    
    #code    = remove_print_function_statements(code)      # remove lines starting with "print(function_name("
    code    = remove_irrelevant_lines(code)                # remove specific irrelevant lines
    if signature:
        is_func = is_function(code)
        if not is_func:                                    # add func signature if missing
            code = join_properly(signature, code)
        else:
            code = merge_imports_with_code(signature, code)    # add all original import statements
    code = add_typing_import(code)                         # add from typing import * (if missing)
    return code.strip()


def join_properly(signature: str, body: str) -> str:
    """
    Join function signature and body using proper indentation.
        :param signature: function signature, something like 'def my_func(x):'.
        :param body: function body.
        :return: A single string that combines the signature and the (re)indented body.
    """
    signature = signature.strip()
    lines = body.splitlines()

    # Strip empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return signature
    
    # Check if first non-empty line starts with space or tab
    first_line = lines[0]
    if first_line[:1] in (' ', '\t'):                              # already indented
        final_lines = [signature] + lines
    else:                                                          # needs indentation
        indentation = '    '
        indented_lines = [indentation + line for line in lines]
        final_lines = [signature] + indented_lines

    return '\n'.join(final_lines)


def get_code_in_fences(code: str) -> str:
    ''' Parse code from 4 different variations of code fences '''
    # edge cases
    if not isinstance(code, str):
        print('WARNING. Not a string passed for evaluation of generated code - returning an empty string')
        return ''
    if not code:
        return code
    
    fence_pattern = re.compile(r"```[ \t]*(?P<lang>python(?:3)?)", re.IGNORECASE)    # shorter, simpler
    #fence_pattern = re.compile(r"```[ \t]*(?P<lang>python(?:3)?)[ \t]*(?=$|\r?\n)", re.IGNORECASE)
    match = fence_pattern.search(code)
    if match:
        fence = match.group(0)
        _, _, remainder = code.partition(fence)        # extract text from end of fence to end of string
        code, _, _ = remainder.partition("```")        # keep only text before next triple backticks        
    return code.strip()


def remove_assert_statements(code: str) -> str:
    ''' Remove lines starting with "assert" '''
    if not 'assert' in code: return code
    return '\n'.join( line for line in code.splitlines() if not line.lstrip().startswith('assert') )


def remove_print_function_statements(code: str) -> str:
    ''' Remove lines starting with "print(function_name(" '''
    
    # find function_name
    match = re.search(r'def\s+(\w+)\(', code)
    if not match: return code
    func_name     = match.group(1)
    print_pattern = re.compile(r'^\s*print\(\s*' + re.escape(func_name) + r'\(')  # matching "print(function_name("

    # if function_name found, remove lines starting with "print(function_name("
    return '\n'.join( line for line in code.splitlines() if not print_pattern.match(line) )


def is_function(text: str) -> bool:
    ''' Is input text a Python function? '''
    lines = text.splitlines()
    for line in lines:
        if line.startswith('def ') or line.startswith('async def ') or line.startswith('class '):
            return True
        elif re.search(r'=\s*lambda', line):
            return True
    return False


def add_typing_import(code):
    ''' If needed add line: from typing import *'''
    typing_import = 'from typing import *'
    if typing_import[:-2] not in code:
        if code.strip().startswith('def'):
            code = typing_import + '\n\n' + code
        else:
            code = typing_import + '\n' + code
    return code


def get_import_statements(code: str) -> str:
    ''' Keeps only lines starting w/'import' or 'from ' that occur before function definition'''
    lines = code.splitlines()
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('def '):          # avoid finding lines starting with from in docstring
            break
        if stripped_line.startswith('import ') or stripped_line.startswith('from '):
            filtered_lines.append( stripped_line )    # these are imports before the function definition => must be stripped
    return '\n'.join(filtered_lines).strip()


def merge_imports_with_code(signature: str, code: str,) -> str:
    '''
      1. Extract import statements from the signature.
      2. Add them to code.
      Simplified version. Earlier version of this function tried to extract any import statements
      from code too to combine the unique import statements from both signature and code.
      But it turned out that this intrusive method is prone to errors.
    '''
    # need not signature here in case signature=None
    if not signature or not signature.strip():
        return code
        
    # Step 1: Extract imports from signature
    signature_imports = get_import_statements(signature)
    if not signature_imports.strip():
        return code
    
    # Step 2: Add them to code
    code = signature_imports + '\n' + code
    return code.strip()


def split_code_on_main(code: str) -> str:
    '''
    Keep only the part of code that comes before the first
    occurrence of 'if __name__ == "__main__":'
    '''
    if not code:
        return code
    markers = ['if __name__ == "__main__":', "if __name__ == '__main__':"]
    positions = [code.find(marker) for marker in markers]
    valid_positions = [pos for pos in positions if pos != -1]
    if valid_positions:
        first_marker_index = min(valid_positions)
        return code[:first_marker_index].strip()
    return code


def remove_irrelevant_lines(code: str) -> str:
    '''
    Remove lines that are strictly one of the following (after strip()):
      - ``` 
      - [code]
      - [/code]
    '''
    if not code:
        return code
    filtered_lines = []
    for line in code.splitlines():
        stripped_line = line.strip()
        if stripped_line in ("```", "[code]", "[/code]"):
            continue
        filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False) -> None:
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
                
                
def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
                    
                    
def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


if __name__ == "__main__":
    
    # unit tests (convert to real tests)
    s = '''
    Here is an improved solution:
    ```Python3
    def get_res(s):
        return s.upper()
    ```
    '''
    print(get_code_in_fences(s))
    print(get_code_in_fences(''))
    print(get_code_in_fences(' '))
    
    s = '''
    def is_lower(s):
        s = s[::-1]
        return s.lower() 

    assert is_lower("InValid") == "invalid"
    assert is_lower("TruE") == "true"
    assert is_lower("SenTenCE") == "sentence"
    '''
    print(remove_assert_statements(s))
    print(remove_assert_statements(''))
    print(remove_assert_statements(' '))
    
    s = """def check_value(dict, value):
        return all(val == value for val in dict.values())

    print(check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},10))
     print(check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},12))
      print(check_value( {'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},5))
        print( check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},5))
    """
    print(remove_print_function_statements(s))
    print(remove_print_function_statements(''))
    print(remove_print_function_statements(' '))
    
    s1 = '''
    def func():
        return
    '''
    s2 = '''
    return
    '''
    for s in [s1, s2]:
        print(is_function(s))
        
    
    s1 = '''
    Here is an improved solution:
    ``` Python3
    def get_res(s):
        return s.upper()
    ```
    '''

    s2 = '''
    The proposed solution is already correct. It defines a Python function named 'is_lower', which takes a string 's' as input and returns the string in lower case. The function uses the 'lower' method of the string class to achieve this. 

    The proposed solution has already been optimally improved, eliminating redundancies and ensuring the code is runnable. Also, the output meets all the given requirements including, the correct implementation of the function, absence of extra text or symbols, and execution-ready code. No additional error corrections or improvements are needed as per your instruction. 

    The proposed solution:

    ```python
    def is_lower(s):
        s = s[::-1]
        return s.lower() 

    assert is_lower("InValid") == "invalid"
    assert is_lower("TruE") == "true"
    assert is_lower("SenTenCE") == "sentence"
    '''

    s3 = """def check_value(dict, value):
        return all(val == value for val in dict.values())

    print(check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},10))
    print(check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},12))
    print(check_value({'Cierra Vega': 12, 'Alden Cantrell': 12, 'Kierra Gentry': 12, 'Pierre Cox': 12},5))
    """

    s4 = '''
    Here is an improved solution:
    ```Python3
    def get_res(s):
        return s.upper()
    ```
    '''

    signature = '''import collections
    import random
    import string

    def task_func(length=100):
        """ Function to perform a task """
    '''
    
    for idx, s in enumerate([s1, s2, s3]):
        print(f'Iteration {idx+1}\n')
        print(clean_code(s), '\n')
        print(clean_code(s, signature), '\n')
        
    code_with_imports = '''import collections
    import random
    import string

    def task_func(length=100):
        return 1
    '''
    print( get_import_statements(code_with_imports) )

    code1 = '''import time
    import datetime

    def task_func(length=100):
        length *= 100
        return 5 + length
    '''

    code2 = '''def task_func(length=100):
        length *= 100
        return 5 + length
    '''

    for code in [code1, code2]:
        print(merge_imports_with_code(signature, code), '\n')
        print(clean_code(code, signature), '\n')
