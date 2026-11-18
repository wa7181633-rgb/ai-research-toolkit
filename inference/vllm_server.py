"""
Advanced vLLM Serving Configuration
===================================
Production-oriented settings for high throughput + good latency.

Key techniques used:
- PagedAttention (default in vLLM)
- FP8 KV cache (major memory win on Hopper+)
- Prefix caching
- Chunked prefill
- Controlled GPU memory utilization
"""

from vllm import LLM, SamplingParams
import argparse

def create_llm(
    model: str,
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.92,
    kv_cache_dtype: str = "fp8",
    enable_prefix_caching: bool = True,
    tensor_parallel_size: int = 1,
):
    """
    Create a high-performance vLLM engine.
    """
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,          # fp8 strongly recommended on H100+
        enable_prefix_caching=enable_prefix_caching,
        trust_remote_code=True,
        # Additional advanced flags (uncomment as needed):
        # enable_chunked_prefill=True,
        # max_num_seqs=256,
        # max_num_batched_tokens=8192,
    )
    return llm


def demo_generate(llm: LLM):
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
    )

    prompts = [
        "Explain Group Relative Policy Optimization (GRPO) and why it is more memory-efficient than PPO.",
        "What are the main benefits of using FP8 KV cache in vLLM?",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n===== Prompt {i+1} =====")
        print(output.prompt)
        print("\n----- Response -----")
        print(output.outputs[0].text)
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    args = parser.parse_args()

    print("Initializing advanced vLLM engine...")
    print(f"Model: {args.model}")
    print(f"Max model length: {args.max_model_len}")
    print(f"KV cache dtype: fp8")
    print(f"Prefix caching: enabled")

    llm = create_llm(
        model=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
    )

    demo_generate(llm)
