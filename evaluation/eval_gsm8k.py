# Evaluation Script (GSM8K Style Benchmark)

import re
from vllm import LLM, SamplingParams

def evaluate_gsm8k():
    llm = LLM(model="unsloth/llama-3-8b-Instruct-bnb-4bit", tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    
    # Sample GSM8K evaluation prompts
    prompts = [
        "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nA:",
        "Q: A robe takes 2 bolts of blue fabric and half as much white fabric. How many bolts of fabric total does it take?\nA:"
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {output.prompt}\nOutput: {generated_text}\n" + "-"*40)

if __name__ == "__main__":
    evaluate_gsm8k()
