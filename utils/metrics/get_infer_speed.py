"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-12-03 20:09:17
❤LastEditTime: 2023-12-05 08:34:35
❤FilePath: get_infer_speed
❤Github: https://github.com/MilknoCandy
"""
import time
import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = now_dir.rsplit('/', 2)[0]
sys.path.append(root_dir)
os.chdir(now_dir)
import pandas as pd
import torch
from network_architecture.network_tools.build_model import build_network
from utils.to_torch import maybe_to_torch, to_cuda
from utils.metrics.fp16_utils import wrap_fp16_model

def compute_throughput(model, bs, input_size, iteration=100):
    """compute model's inference time(cpu and gpu) and FPS(Frames Per Second)

    Args:
        model (_type_): models already sotred on GPU
        input_size (_type_): input's shape
        iteration (int, optional): _description_. Defaults to 100.
    """
    # 切换为验证模式
    model.eval()

    input = torch.ones(()).new_empty(
        (bs, *input_size),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    #############################GPU INFERENCE TIME##############################
    input = to_cuda(input)          # 放到GPU上

    # GPU预热, GPU原本可能处在休眠状态
    for _ in range(10):
        model(input)
    
    # 设置GPU事件
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    with torch.no_grad():
        start.record()
        for _ in range(iteration):
            model(input)
        stop.record()
        stop.synchronize()
        elapsed_time_gpu = start.elapsed_time(stop)         # ms
        bs_time_gpu = elapsed_time_gpu / iteration
        thorughput_gpu = iteration * bs / elapsed_time_gpu * 1000
        torch.cuda.empty_cache()
    #############################CPU INFERENCE TIME##############################
    with torch.no_grad():
        model = model.cpu()
        input = input.cpu()
        # t_start = time.perf_counter()
        t_start = time.time()
        for _ in range(iteration):
            model(input)
        # elapsed_time_cpu = time.perf_counter() - t_start            # s
        elapsed_time_cpu = time.time() - t_start            # s
        bs_time_cpu = elapsed_time_cpu / iteration * 1000
        thorughput_cpu = iteration * bs / elapsed_time_cpu

    return bs_time_gpu, bs_time_cpu, thorughput_gpu, thorughput_cpu

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # models = ['unext', 'ca-net', 'shformer_add', 'shformer_flash', 'shformer_flash_d2', 'shformer_pvt']
    models = ['shformer_add_MLA', 'shformer_add_SeD', 'shformer_add_PLD']
    # models = ['shformer_add']
    # bs_lab = ['b9', 'b10']
    bs_lab = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
    bs = [1, 2, 3, 4, 5, 6, 7, 8]
    fp16 = True
    
    model_latency_gpu_df = pd.DataFrame(index=bs_lab)
    model_latency_cpu_df = pd.DataFrame(index=bs_lab)
    model_throughput_gpu_df = pd.DataFrame(index=bs_lab)
    model_throughput_cpu_df = pd.DataFrame(index=bs_lab)
    for model_name in models:
        bs_latency_gpu = []
        bs_latency_cpu = []
        bs_throughput_gpu = []
        bs_throughput_cpu = []
        for b in bs:
            print('#'*10, ' Batch Size is ', '[', b, ']', ' ', '#'*10)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            print("=> creating model %s" % model_name)
            # model = build_network(model_name).to(device)
            model = build_network(model_name)
            if fp16:
                # model = model.half()
                wrap_fp16_model(model)
            model = model.cuda()
            
            speed_metrics = compute_throughput(model, bs=b, input_size=(3, 512, 512), iteration=100)
            bs_latency_gpu.append(speed_metrics[0])
            bs_latency_cpu.append(speed_metrics[1])
            bs_throughput_gpu.append(speed_metrics[2])
            bs_throughput_cpu.append(speed_metrics[3])
        model_latency_gpu_df[model_name] = bs_latency_gpu
        model_latency_cpu_df[model_name] = bs_latency_cpu
        model_throughput_gpu_df[model_name] = bs_throughput_gpu
        model_throughput_cpu_df[model_name] = bs_throughput_cpu
    model_latency_gpu_df.T.to_excel('./model_latency_gpu.xlsx')
    model_latency_cpu_df.T.to_excel('./model_latency_cpu.xlsx')
    model_throughput_gpu_df.T.to_excel('./model_throughput_gpu.xlsx')
    model_throughput_cpu_df.T.to_excel('./model_throughput_cpu.xlsx')
