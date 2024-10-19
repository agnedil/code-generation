import json
import openai
import subprocess

# Replace 'your-api-key-here' with your actual OpenAI API key
openai.api_key = 'your-api-key-here'

def load_humaneval_dataset(file_path):
    tasks = []
    with open(file_path, 'r') as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    return tasks


def get_completion(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a coding assistant. You will be provided with a function signature and docstring. Your task is to implement the function body to correctly solve the problem described in the docstring. Do not include any additional explanation or text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
        stop=["\n\n"]
    )
    completion = response.choices[0].message.content
    return completion


def execute_code(code):
    try:
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, '', 'TimeoutExpired'


def main():
    tasks = load_humaneval_dataset('human_eval.jsonl')
    total_tasks = len(tasks)
    passed_tasks = 0

    for task in tasks:
        task_id = task['task_id']
        prompt = task['prompt']
        canonical_solution = task['canonical_solution']
        test_code = task['test']

        print(f"Processing task {task_id}")

        completion = get_completion(prompt)

        # Combine prompt and completion to get candidate code
        candidate_code = prompt + completion

        # Combine candidate code and test code
        full_code = candidate_code + '\n' + test_code

        returncode, stdout, stderr = execute_code(full_code)

        if returncode == 0:
            print(f"Task {task_id} passed.")
            passed_tasks += 1
        else:
            print(f"Task {task_id} failed.")
            print(f"Stderr: {stderr}")

    print(f"Total tasks: {total_tasks}, Passed tasks: {passed_tasks}")

if __name__ == '__main__':
    main()
