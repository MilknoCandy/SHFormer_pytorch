"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-15 15:34:14
❤LastEditTime: 2023-12-04 10:08:25
❤Github: https://github.com/MilknoCandy
"""
import time
import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = now_dir.rsplit('/', 2)[0]
sys.path.append(root_dir)
import pandas as pd
import torch
from network_architecture.network_tools.build_model import build_network
from utils.to_torch import maybe_to_torch, to_cuda

def compute_speed_mmcv(model, bs, input_size, iteration=100):
    num_warmup = 50
    pure_inf_time = 0
    total_iters = 200
    batch_size = bs
    for i in range(total_iters):
        sample = torch.ones(()).new_empty(
            (batch_size, *input_size),
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            model(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % 50 == 0:
                fps = (i + 1 - num_warmup) * batch_size / pure_inf_time
                # thr = (i + 1 - num_warmup) / pure_inf_time
                thr = pure_inf_time / (i + 1 - num_warmup) * 1000
                print('Done image [{:3}/ {}], '.format(i+1, total_iters) + 
                      'fps: {:.2f} img / s'.format(fps) + ', infer time: {:.3f} ms / bs'.format(thr))
                # print('Done image [{:3}/ {}], '.format(i+1, total_iters) + 
                #       'fps: {:.2f} img / s'.format(fps) + ', throughput: {:.3f} bs / s'.format(thr))

        if (i + 1) == total_iters:
            fps = (total_iters - num_warmup) * batch_size / pure_inf_time
            thr = (total_iters - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s, throughput: {thr:.3f} bs / s')
            # print('Overall fps: {:.2f} img / s'.format(fps))
            break
        
def compute_speed(model, input_size, iteration=100):
    """compute model's inference time(cpu and gpu) and FPS(Frames Per Second)

    Args:
        model (_type_): models already sotred on GPU
        input_size (_type_): input's shape
        iteration (int, optional): _description_. Defaults to 100.
    """
    # 切换为验证模式
    model.eval()

    input = torch.randn(*input_size)
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
        speed_time_gpu = elapsed_time_gpu / iteration
        fps_gpu = iteration / elapsed_time_gpu * 1000
        torch.cuda.empty_cache()
    #############################CPU INFERENCE TIME##############################
    # with torch.no_grad():
    #     model = model.cpu()
    #     input = input.cpu()
    #     # t_start = time.perf_counter()
    #     t_start = time.time()
    #     for _ in range(iteration):
    #         model(input)
    #     # elapsed_time_cpu = time.perf_counter() - t_start            # s
    #     elapsed_time_cpu = time.time() - t_start            # s
    #     speed_time_cpu = elapsed_time_cpu / iteration * 1000

    print('Elapsed Time(GPU): [%.2f ms / %d iter]' % (elapsed_time_gpu, iteration))
    print('Speed Time(GPU): %.2f ms / iter   FPS: %.2f' % (speed_time_gpu, fps_gpu))
    # print('Speed Time(CPU): %.2f ms / iter' % speed_time_cpu)
    # return speed_time_gpu, fps_gpu, speed_time_cpu
    return speed_time_gpu, fps_gpu

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # models = ['unext', 'ca-net', 'shformer_add', 'shformer_flash', 'shformer_flash_d2', 'shformer_pvt']
    models = ['shformer_add']
    
    model_latency_gpu_df = {}
    model_latency_cpu_df = {}
    model_fps_gpu_df = {}
    for model_name in models:
        model_latency_gpu = []
        model_latency_cpu = []
        model_fps_gpu = []
        # model_name = "shformer_pvt"
        # model_name = "unext"
        # for b in [1, 2, 4, 6, 8, 16]:
            
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("=> creating model %s" % model_name)
        model = build_network(model_name).to(device)
        
        # gpu_latency, gpu_fps, cpu_latency = compute_speed(model, (7, 3, 512, 512), iteration=100)
        gpu_latency, gpu_fps = compute_speed(model, (7, 3, 512, 512), iteration=100)
        # model_latency_gpu.append(gpu_latency)
        # model_latency_cpu.append(cpu_latency)
        # model_fps_gpu.append(gpu_fps)
        # compute_speed_mmcv(model, 7, (3, 512, 512), iteration=100)
