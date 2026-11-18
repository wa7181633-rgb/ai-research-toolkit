# Advanced vLLM Serving Configuration & Script

from vllm import LLM, SamplingParams

def run_vllm_server_demo():
    # Initialize vLLM with PagedAttention and FP8 optimizations
    llm = LLM(
        model="unsloth/llama-3-8b-Instruct-bnb-4bit",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        tensor_parallel_size=1
    )
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
    prompts = ["Explain Group Relative Policy Optimization (GRPO) in simple terms."]
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Generated Response:\n{output.outputs[0].text}")

if __name__ == "__main__":
    run_vllm_server_demo()
