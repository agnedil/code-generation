import json
import openai
import os
import sys
import time
import traceback

# Retrieve the API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    print("Error: The OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

def load_mbpp_dataset(file_path):
    """Load tasks from the MBPP dataset."""
    tasks = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                task = json.loads(line)
                tasks.append(task)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
    return tasks

def get_completion(prompt, model="gpt-4"):
    """Get code completion from the OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that writes Python functions. "
                        "Given a problem description, write a correct and efficient Python function "
                        "that solves the problem. Do not include any additional text or explanations."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            stop=["\n\n"]
        )
        completion = response.choices[0].message.content
        return completion.strip()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None

def execute_code(code, test_cases, timeout=5):
    """
    Execute the given code with the provided test cases in a controlled environment.

    WARNING: Executing code can be dangerous. This function should be used
    with caution and ideally in a secure, sandboxed environment.
    """
    # Prepare a restricted execution environment
    exec_globals = {'__builtins__': {}}
    exec_locals = {}
    try:
        # Compile the candidate code to detect syntax errors
        compiled_code = compile(code, filename="<string>", mode="exec")
        # Execute the candidate code
        exec(compiled_code, exec_globals, exec_locals)
        # Now run each test case
        for test in test_cases:
            # Each test is a string of code to execute
            test_code = compile(test, filename="<string>", mode="exec")
            exec(test_code, exec_globals, exec_locals)
        return True, exec_locals
    except Exception as e:
        # Capture the traceback
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        return False, traceback_str

def main():
    dataset_path = 'mbpp.jsonl'
    tasks = load_mbpp_dataset(dataset_path)
    total_tasks = len(tasks)
    passed_tasks = 0

    for task in tasks:
        task_id = task.get('task_id', 'unknown')
        prompt = task['text']
        test_list = task['test_list']
        code_solution = task['code_solution']  # The reference solution

        print(f"\nProcessing task {task_id}")

        completion = get_completion(prompt)

        if completion is None:
            print(f"Skipping task {task_id} due to API error.")
            continue

        candidate_code = completion

        # Run the test cases
        passed, result = execute_code(candidate_code, test_list)

        if passed:
            print(f"Task {task_id} passed.")
            passed_tasks += 1
        else:
            print(f"Task {task_id} failed.")
            print(f"Error:\n{result}")

        # Rate limiting to avoid hitting API rate limits
        time.sleep(1)  # Adjust the sleep time as needed

    print(f"\nTotal tasks: {total_tasks}, Passed tasks: {passed_tasks}")

if __name__ == '__main__':
    main()
