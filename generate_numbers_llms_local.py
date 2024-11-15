import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import datetime
from openai import OpenAI
from tqdm import tqdm
import time
import pandas as pd

# Create necessary directories if they don't exist
data_dir = "./data"
os.makedirs(f"{data_dir}/runs", exist_ok=True)

# run timestamp
run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed" # parameter required but value not used
)

def prompt_oai(prompt: str, attempts: int=3, model: str=None) -> str:
    """
    Send prompt to local LLM server.

    Parameters
    ----------
    prompt : str
        The prompt to send to the model
    attempts : int, default=3
        Number of retry attempts
    model : str, default=None
        Model identifier to use

    Returns
    -------
    str
        The model's response, or None if all attempts fail
    """
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=model or "gemma-2-9b-it",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=-1
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in prompt_oai (attempt {attempt + 1}/{attempts}): {e}")
            
            sleep_time = (2 ** attempt) * 2
            print(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
            
            if attempt == attempts - 1:
                return None
            continue


def is_valid_response(response: str, range_min: int, range_max: int) -> bool:
    return range_min <= response <= range_max

def prompt_oai_para(prompts: list, max_workers=5, desc: str=None, model: str="gpt-4o") -> list:
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(prompt_oai, prompt, model=model): prompt for prompt in prompts}
        
        for future in as_completed(futures):
            prompt = futures[future]
            try:
                result = future.result()
                if result is None:
                    response = None
                else:
                    try:
                        response = int(result.strip())
                    except ValueError:
                        response = None
                results.append({
                    'prompt': prompt,
                    'response': response,
                    'raw_response': result
                })
            except Exception as e:
                print(f"Error in prompt_oai_para: {e}")
                results.append({
                    'prompt': prompt,
                    'response': None,
                    'raw_response': str(e)
                })
    
    # Ensure results are in same order as input prompts
    sorted_results = []
    prompt_to_result = {r['prompt']: r for r in results}
    for prompt in prompts:
        sorted_results.append(prompt_to_result[prompt])
    
    return sorted_results

def generate_random_range():
    """Generate a pair of numbers, returned sorted from low to high, with at least 10 digits between them, but no more than 500 between them."""
    rng = sorted(random.sample(range(-1000, 1000), 2))
    if 10 < (rng[1] - rng[0]) < 500:
        return rng
    return generate_random_range()

PROMPTS = 5

ranges = [
    (0, 100),
    (0, 10),

    # randomly generated but hardcoded for testing
    (-745, -556),
    (-359, 39),
    (328, 508),
    (-647, 489),
    (-863, 557),

    # # add some randomly generated ranges
    # *[generate_random_range() for _ in range(PROMPTS)]
]

prompts_info = []
for rng in ranges:
    prompts_info.append({
        'prompt': f'Generate a random number between {rng[0]} and {rng[1]}. It must be a whole number. Return only the number and nothing else, as it will be parsed by Python immediately downstream.',
        'range': rng
    })

for prompt in prompts_info:
    print(prompt['prompt'])

models = [
    # "gemma-2-2b-it",
    "gemma-2-9b-it",
    # "llama-3.2-3b-instruct"
    # Add any other local models you want to test via LMStudio
]

RUNS = 200  # Number of times to run the same prompts
RESULTS = []

run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
total_iterations = RUNS * len(models)

# beginning runs printout
print(f"\n\nRunning {total_iterations} iterations")
print("-"*50)

with tqdm(total=total_iterations, desc="Processing runs") as pbar:
    for run_idx in range(RUNS):
        for model in models:
            results = prompt_oai_para(
                prompts=[info['prompt'] for info in prompts_info],
                max_workers=5, 
                model=model
            )

            # add model name, prompt, and range info to results
            for info, result in zip(prompts_info, results):
                if result['response'] is not None:
                    # Validate prompt matching
                    assert result['prompt'] == info['prompt'], "Prompt mismatch detected!"
                    
                    is_valid = is_valid_response(result['response'], info['range'][0], info['range'][1])
                    RESULTS.append({
                        'run': run_idx,
                        'generation_method': model,
                        'prompt': info['prompt'],
                        'range_min': info['range'][0],
                        'range_max': info['range'][1],
                        'response': result['response'],
                        'raw_response': result['raw_response'],
                        'is_valid': is_valid
                    })
            pbar.update(1)

        # write df to disk every 100 runs and after last run
        if (run_idx + 1) % 100 == 0 or run_idx == RUNS - 1:
            results_df = pd.DataFrame(RESULTS)
            results_df.to_csv(f"{data_dir}/runs/{run_timestamp}-{RUNS}.csv", index=False)

print()

# Add analysis code
model_accuracy = {}
for result in RESULTS:
    model = result['generation_method']
    if model not in model_accuracy:
        model_accuracy[model] = {'valid': 0, 'total': 0}
    
    model_accuracy[model]['total'] += 1
    if result['is_valid']:
        model_accuracy[model]['valid'] += 1

# Print accuracy stats
for model, stats in model_accuracy.items():
    accuracy = (stats['valid'] / stats['total']) * 100
    print(f"{model}: {accuracy:.1f}% accurate ({stats['valid']}/{stats['total']})")

results_df = pd.DataFrame(RESULTS)
results_df.to_csv(f"{data_dir}/runs/{run_timestamp}-{RUNS}.csv", index=False)

# count the number of runs where each model was the most accurate
for model in results_df['generation_method'].unique():
    print(f"{model}: {results_df[results_df['generation_method'] == model]['is_valid'].value_counts(normalize=True)}")
