##### HUMAN EVAL AND BIG CODE BENCH PROMPTS #####

complete_code_prompt_basic = '''
Complete the following Python code:
{}
'''

complete_code_prompt = '''
1. Act as an experienced Python software developer and complete the code provided below.
2. Enclose your output code in the following code fences:
```python
[code]
```
3. Complete the following Python code:
{}
'''

complete_code_prompt_full = '''
1. Act as an experienced Python software developer and complete the starter code provided below.
2. Understand the task described in the starter code, generate the completion, and integrate the completion with the starter code using Python's correct syntax and proper indentation.
3. Stop generating immediately after the return statement or the final line of the function.
4. Exclude any non-code content. For example, exclude explanatory text, exclude example usage, exclude test cases, exclude phrases like "Completion:" or "Here is a completion", exclude any other headings, and exclude anything that can be considered as non-executable Python code.
5. Your final output must be only code that can be executed directly by a Python interpreter without errors.
6. Enclose your final output code in the following code fences:
```python
[code]
```
7. Using the above instructions, complete the following Python code:
{}
'''

complete_code_prompt_full_OLD = '''
1. Act as an experienced Python software developer and complete the starter code provided below.
2. Make sure you understand the starter code, generate its completion, and integrate your completion with the provided starter code using Python's correct syntax and proper indentation.
3. Your output must be only code and it must not contain any irrelevant non-code content: it should be with no explanatory text, no example usage, no test cases, no phrases like "Completion:" or "Here is a completion", and no other headings.
4. Therefore, stop generating immediately after the return statement or the final line of the function.
5. Your final output code must be directly runnable by a Python interpreter without errors.
6. Enclose your final runnable output code in the following code fences:
```python
[code]
```
7. Using all of the above instructions, complete the following Python code:
{}
'''

##### MBPP AND LBPP PROMPTS #####

complete_task_prompt_basic = '''
Complete the following task:
{}.

Your output code must satisfy these tests:
{}
'''

complete_task_prompt = '''
1. Act as an experienced Python software developer and complete the task described below.
2. Enclose your output code in the following code fences:
```python
[code]
```
3. Complete the following task:
{}

4. Verify that your output code satisfies the following tests, but do not include the tests in the output code:
{}
'''


complete_task_prompt_full = '''
1. Act as an experienced Python software developer.
2. Understand the task described below and generate an efficient Python solution that successfully passes all the tests listed below.
3. Stop generating immediately after the return statement or the final line of the function.
4. Exclude any non-code content. For example, exclude explanatory text, exclude example usage, exclude test cases, exclude phrases like "Solution:" or "Here is a solution", exclude any other headings, and exclude anything that can be considered as non-executable Python code.
5. Your final output must be only code that can be executed directly by a Python interpreter without errors.
6. Enclose your final output code in the following code fences:
```python
[code]
```
7. Using the above instructions, complete the following task:
{}

8. Verify that your output code satisfies the following tests, but do not include the tests in the output code:
{}
'''

complete_task_prompt_full_OLD = '''
1. Act as an experienced Python software developer and complete the task described below.
2. Make sure you understand the task and generate a solution using Python's correct syntax and proper indentation.
3. Your output must be only code and it must not contain any irrelevant non-code content: it should be with no explanatory text, no example usage, no test cases, no phrases like "Solution:" or "Here is a solution", and no other headings.
4. Therefore, stop generating immediately after the return statement or the final line of the function.
5. Your final output code must be directly runnable by a Python interpreter without errors.
6. Enclose your final runnable output code in the following code fences:
```python
[code]
```
7. Using all of the above instructions, complete the following task:
{}

8. Verify that your output code satisfies the following tests, but do not include the tests in the output code:
{}
'''

reflection_prompt_basic = '''1. Read and understand the REQUIREMENTS and the PROPOSED SOLUTION listed below.
2. Correct and improve the PROPOSED SOLUTION.

REQUIREMENTS:
{}

PROPOSED SOLUTION:
{}'''

reflection_prompt  = '''1. Carefully read and understand the REQUIREMENTS and the PROPOSED SOLUTION listed below.
2. Correct any errors in the PROPOSED SOLUTION and improve it as much as possible.
3. Enclose the new improved solution in the following code fences:
```python
[code]
```

REQUIREMENTS:
{}

PPROPOSED SOLUTION:
{}'''

reflection_prompt_full  = '''+++Refine(iterations=3)
* Carefully read and understand the REQUIREMENTS and the PROPOSED SOLUTION listed below.
* Correct all errors in the logic, syntax, or any other aspects of the PROPOSED SOLUTION. Remove any redundant code.
* Make sure the new improved solution satisfied the REQUIREMENTS in the best possible way.

+++Output Criteria
* Stop generating the new improved solution immediately after the return statement or the final line of the function.
* Exclude any non-code content. For example, exclude explanatory text, exclude example usage, exclude test cases, exclude phrases like "Solution:" or "Here is a solution", exclude any other headings, and exclude anything that can be considered as non-executable Python code.
* Your final output must be only code that can be executed by a Python interpreter without errors.

+++Output Format
* Enclose the new improved solution in the following code fences:
```python
[code]
```

+++REQUIREMENTS:
{}

+++PROPOSED SOLUTION:
{}'''


if __name__ == "__main__":
    print(complete_code_prompt_basic.format('Next line'))