import json
import openai
import os
import sys
import traceback
import time

# Retrieve the API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    print("Error: The OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

def load_humaneval_dataset(file_path):
    """Load tasks from the HumanEval dataset."""
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
                        "You are a coding assistant. You will be provided with a function "
                        "signature and docstring. Your task is to implement the function body "
                        "to correctly solve the problem described in the docstring. Do not include "
                        "any additional explanation or text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=512,
            stop=["\nclass", "\ndef", "\nif", "\n#"]
        )
        completion = response.choices[0].message.content
        return completion.strip()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None

def execute_code(code, timeout=5):
    """
    Execute the given code in a controlled environment.

    WARNING: Executing code can be dangerous. This function should be used
    with caution and ideally in a secure, sandboxed environment.
    """
    # Prepare a restricted execution environment
    exec_globals = {'__builtins__': {}}
    exec_locals = {}
    try:
        # Record the start time
        start_time = time.time()
        # Compile the code to detect syntax errors
        compiled_code = compile(code, filename="<string>", mode="exec")
        # Execute the compiled code
        exec(compiled_code, exec_globals, exec_locals)
        # Check for timeout
        if time.time() - start_time > timeout:
            return False, "Execution timed out."
        return True, exec_locals
    except Exception as e:
        # Capture the traceback
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        return False, traceback_str

def run_tests(candidate_code, test_code):
    """Run the test code against the candidate code."""
    full_code = candidate_code + '\n' + test_code
    passed, result = execute_code(full_code)
    return passed, result

def main():
    dataset_path = 'human_eval.jsonl'
    tasks = load_humaneval_dataset(dataset_path)
    total_tasks = len(tasks)
    passed_tasks = 0

    for task in tasks:
        task_id = task['task_id']
        prompt = task['prompt']
        test_code = task['test']

        print(f"\nProcessing task {task_id}")

        completion = get_completion(prompt)

        if completion is None:
            print(f"Skipping task {task_id} due to API error.")
            continue

        # Combine prompt and completion to get candidate code
        candidate_code = prompt + completion

        # Run the test code
        passed, result = run_tests(candidate_code, test_code)

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
