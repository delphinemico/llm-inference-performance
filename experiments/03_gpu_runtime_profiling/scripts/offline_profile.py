import argparse
import torch
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--workload", choices=["prefill", "decode"], required=True)
args = parser.parse_args()

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enable_prefix_caching=False,
)

if args.workload == "prefill":
    prompt = ("Explain the following enterprise AI context carefully. " * 320)
    max_tokens = 64
else:
    prompt = ("Summarize this AI serving scenario. " * 80)
    max_tokens = 512

sampling = SamplingParams(
    temperature=0,
    max_tokens=max_tokens,
    min_tokens=max_tokens if args.workload == "decode" else 0,
    ignore_eos=args.workload == "decode",
)

# Warm the exact workload before profiling so one-time JIT work is excluded.
llm.generate([prompt], sampling)
torch.cuda.synchronize()

cudart = torch.cuda.cudart()
cudart.cudaProfilerStart()

outputs = llm.generate([prompt], sampling)

torch.cuda.synchronize()
cudart.cudaProfilerStop()

output = outputs[0]

print({
    "workload": args.workload,
    "prompt_tokens": len(output.prompt_token_ids),
    "completion_tokens": len(output.outputs[0].token_ids),
})
