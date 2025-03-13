##### HUMAN EVAL AND BIG CODE BENCH PROMPTS #####

complete_code_prompt_basic = 'Complete the following Python code:\n{}'

complete_code_prompt = '''
**Role**
You are a Python software programmer.

**Instructions**
As a programmer, you are required to complete the following Python code.

**Code Format**
Output the completed code in the following format:
```python
<code>
```

**Code To Be Completed**
{}
'''

complete_code_prompt_full = '''
**Role**
You are a Python software programmer.

**Instructions**
1. As a programmer, you are required to complete the code for the function described below.
2. Make sure you understand the function description, complete the code and merge the function description with its completion using the correct Python syntax and proper indentation.
3. Your output should be with no example usage and no test cases - stop generating immediately after the return statement or the final line of the function.
4. Output only the complete, runnable Python code with no phrases like "Completion:" or "Here is a completion" and no other headings.
5. Your output should be with no explanatory text and no other irrelevant content - it should be directly runnable by a Python interpreter without errors.

**Code Format**
Output the completed code in the following format:
```python
<code>
```

**Function Description**
{}
'''

##### MBPP AND LBPP PROMPTS #####

complete_task_prompt_basic = '''
Complete the following task:
{}.

Your code should satisfy these tests:
{}
'''

complete_task_prompt = '''
**Role**
You are a Python software programmer.

**Instructions**
As a programmer, you are required to complete the following task:
{}

**Code Format**
Output your code in the following format:
```python
<code>
```

Your ouput code should satisfy these tests:
{}
'''


complete_task_prompt_full = '''
**Role**
You are a Python software programmer.

**Instructions**:
1. As a programmer, you are required to complete the task described below.
2. Make sure you understand the task and solve it using the correct Python syntax and proper indentation.
3. Your output should be with no example usage and no test cases - stop generating immediately after the return statement or the final line of the function.
4. Output only the complete, runnable Python code with no phrases like "Solution:" or "Here is a solution" and no other headings.
5. Your output should be with no explanatory text and no other irrelevant content - it should be directly runnable by a Python interpreter without errors.

**Code Format**
Output your code in the following format:
```python
<code>
```

**Task Description**
{}

Your ouput code should satisfy these tests:
{}
'''



if __name__ == "__main__":
    print(human_eval_completion_prompt + 'Next line')