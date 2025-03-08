import re
import sys
import os
import gzip
import json
from typing import Iterable, Dict


def clean_code(code: str, signature: str=None) -> str:
    '''
    Clean code by:
        * Extracting only code between "```python" and "```" code fences.
          Other opening code fences include: "```python3", "```Python", or "```Python3".
        * Remove any lines w/test cases starting with 'assert'
        * Remove any lines w/test cases starting with "print(function_name(".
    '''
    # edge cases
    if not isinstance(code, str):
        print('WARNING. Not a string passed for evaluation of generated code - returning an empty string')
        return ''
    if not code:
        return code    
    code    = get_code_in_fences(code)                     # parse code out of code fences    
    code    = remove_assert_statements(code)               # remove lines starting with 'assert'    
    code    = remove_print_function_statements(code)       # remove lines starting with "print(function_name("
    is_func = is_function(code)
    if signature and not is_func:                          # add func signature if missing
        code = join_properly(signature, code)
        is_func = True
    if is_func and 'from typing import' not in code:
        code = 'from typing import *\n' + code
    return code


def join_properly(signature: str, body: str) -> str:
    """
    Join function signature and body using proper indentation.
        :param signature: function signarure, something like 'def my_func(x):'.
        :param body: function body.
        :return: A single string that combines the signature and the (re)indented body.
    """
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
    if not 'assert' in code: return code.strip()
    return '\n'.join(
                line for line in code.splitlines() if not line.lstrip().startswith('assert')
            ).strip()


def remove_print_function_statements(code: str) -> str:
    ''' Remove lines starting with "print(function_name(" '''
    
    # find function_name
    match = re.search(r'def\s+(\w+)\(', code)
    if not match: return code.strip()
    func_name     = match.group(1)
    print_pattern = re.compile(r'^\s*print\(\s*' + re.escape(func_name) + r'\(')  # matching "print(function_name("

    # if function_name found, remove lines starting with "print(function_name("
    return '\n'.join(
                line for line in code.splitlines() if not print_pattern.match(line)
            ).strip()


def is_function(text: str) -> bool:
    ''' Is input text a Python function? '''
    lines = text.splitlines()
    for line in lines:
        if line.startswith('def ') or line.startswith('async def '):
            return True
    return False


def get_import_statements(code: str) -> str:
    ''' Keeps only lines starting w/'import' or 'from ' '''
    lines = code.splitlines()
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('import ') or stripped_line.startswith('from '):
            filtered_lines.append(line)
    res = '\n'.join(filtered_lines) + '\n'
    s = 'from typing import *'
    if 'from typing import' not in res:
        res = s + '\n' + res
    return res


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
    
    for s in [s1, s2, s3]:
        print(clean_code(s), '\n')
        
    code_with_imports = '''import collections
    import random
    import string

    def task_func(length=100):
        return 1
    '''
    print( get_import_statements(code_with_imports) )


