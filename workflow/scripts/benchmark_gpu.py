"""
Benchmarking CPU, Single GPU, Multi-GPU Performance, and CT Scan Model Training

This script measures performance across different devices for matrix multiplication and simulates a
realistic CT scan training scenario using a 3D convolutional neural network.

### Workflow:
1. **Matrix Size:** Simulates CT scan data using a 3D tensor (512³) for a realistic workload.
2. **GPU Detection:** Checks the number of available GPUs.
3. **Single GPU Benchmark:** Runs matrix multiplication on one GPU (if available).
4. **Multi-GPU Benchmark:** Uses two GPUs for computation if multiple GPUs are detected.
5. **CPU Benchmark:** Runs the same operation on the CPU.
"""

import time

import pandas as pd
import torch
from rich import print  # noqa: A004
from rich.markdown import Markdown

# Matrix Size (Simulating CT Scans)
DIM_SIZE = 512 * 2
SIZE = (DIM_SIZE, DIM_SIZE, DIM_SIZE)  # 3D tensor size
ITERATIONS = 10  # Number of matrix multiplications per benchmark

# Detect available GPUs
num_gpus = torch.cuda.device_count()
print(f'[bold cyan]Number of GPUs available: {num_gpus}[/bold cyan]')


def benchmark(device: str, x: torch.Tensor, y: torch.Tensor) -> float:
	"""Performs matrix multiplication and returns execution time."""
	start = time.time()
	for _ in range(ITERATIONS):
		_ = torch.matmul(x, y)
	torch.cuda.synchronize() if 'cuda' in device else None
	return time.time() - start


# Single GPU Benchmark
gpu_time = None
if num_gpus > 0:
	print('[bold yellow]Benchmarking Single GPU...[/bold yellow]')
	x_gpu, y_gpu = (
		torch.randn(SIZE, device='cuda:0'),
		torch.randn(SIZE, device='cuda:0'),
	)
	gpu_time = benchmark('cuda:0', x_gpu, y_gpu)
	print(f'[green]Single GPU computation: {gpu_time:.3f} sec[/green]')
	del x_gpu, y_gpu
	torch.cuda.empty_cache()

# Multi-GPU Benchmark
multi_gpu_time = None
if num_gpus > 1:
	print('[bold yellow]Benchmarking Multi-GPU...[/bold yellow]')
	x_gpu0, y_gpu1 = (
		torch.randn(SIZE, device='cuda:0'),
		torch.randn(SIZE, device='cuda:1'),
	)
	multi_gpu_time = benchmark('cuda:1', x_gpu0.to('cuda:1'), y_gpu1)
	print(f'[green]Multi-GPU computation: {multi_gpu_time:.3f} sec[/green]')
	del x_gpu0, y_gpu1
	torch.cuda.empty_cache()

# CPU Benchmark
print('[bold yellow]Benchmarking CPU...[/bold yellow]')
x_cpu, y_cpu = torch.randn(SIZE, device='cpu'), torch.randn(SIZE, device='cpu')
cpu_time = benchmark('cpu', x_cpu, y_cpu)
print(f'[green]CPU computation: {cpu_time:.3f} sec[/green]')

del x_cpu, y_cpu

# Compile results into DataFrame
data = {
	'Device': ['GPU', 'Multi-GPU', 'CPU'],
	'Time (s)': [gpu_time, multi_gpu_time, cpu_time],
	'Speedup vs CPU': [
		f'{(cpu_time / t):.2f}x' if t else 'N/A'
		for t in [gpu_time, multi_gpu_time, cpu_time]
	],
}
df = pd.DataFrame(data)  # noqa: PD901

# Display results using rich markdown
print(Markdown(df.to_markdown()))
