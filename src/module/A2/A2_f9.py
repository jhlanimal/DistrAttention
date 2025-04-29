import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from torch.cuda import Stream

from angular import AngularLSHTriton

import torch.utils.benchmark as benchmark
from torch.backends.cuda import sdp_kernel, SDPBackend
import numpy as np
import matplotlib.pyplot as plt
backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

def Flash():
    y_32_0 = []
    y_64_0 = []
    y_128_0 = []
    M = [2048, 2096 ,8192, 10240, 20480, 40960]
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 32)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 32)
        k = np.random.rand(1, 10, M[i], 32)
        v = np.random.rand(1, 10, M[i], 32)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_32(q,k,v):
            o = lsh.flashattn_v2_triton(q,k,v)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_32, q, k, v)
        y_32_0.append(time_in_microseconds)
    
    
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 64)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 64)
        k = np.random.rand(1, 10, M[i], 64)
        v = np.random.rand(1, 10, M[i], 64)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_64(q,k,v):
            o = lsh.flashattn_v2_triton(q,k,v)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_64, q, k, v)
        y_64_0.append(time_in_microseconds)
    
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 128)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 128)
        k = np.random.rand(1, 10, M[i], 128)
        v = np.random.rand(1, 10, M[i], 128)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v):
            o = lsh.flashattn_v2_triton(q,k,v)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v)
        y_128_0.append(time_in_microseconds)
    return y_32_0,y_64_0,y_128_0
    pass

def Ours_sapmele2():
    y_32_1_2 = []
    y_64_1_2 = []
    y_128_1_2 = []
    M = [2048, 2096 ,8192, 10240, 20480, 40960]
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 32)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 32)
        k = np.random.rand(1, 10, M[i], 32)
        v = np.random.rand(1, 10, M[i], 32)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v,ty):
            o = lsh.hash_triton(q,k,v,ty)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v,'2_32')
        y_32_1_2.append(time_in_microseconds)
    
    
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 64)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 64)
        k = np.random.rand(1, 10, M[i], 64)
        v = np.random.rand(1, 10, M[i], 64)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v,ty):
            o = lsh.hash_triton(q,k,v,ty)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v,'2_64')
        y_64_1_2.append(time_in_microseconds)
    
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 128)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 128)
        k = np.random.rand(1, 10, M[i], 128)
        v = np.random.rand(1, 10, M[i], 128)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v,ty):
            o = lsh.hash_triton(q,k,v,ty)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v,'2_128')
        y_128_1_2.append(time_in_microseconds)
    return y_32_1_2,y_64_1_2,y_128_1_2

def Ours_sapmele4():
    #y_32_1_4 = []
    y_64_1_4 = []
    y_128_1_4 = []
    M = [2048, 2096 ,8192, 10240, 20480, 40960]
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 64)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 64)
        k = np.random.rand(1, 10, M[i], 64)
        v = np.random.rand(1, 10, M[i], 64)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v,ty):
            o = lsh.hash_triton(q,k,v,ty)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v,'4_64')
        y_64_1_4.append(time_in_microseconds)
    
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 128)).to(device='cuda', dtype=torch.float16)
    for i in range(len(M)):
        q = np.random.rand(1, 10, M[i], 128)
        k = np.random.rand(1, 10, M[i], 128)
        v = np.random.rand(1, 10, M[i], 128)
        q = torch.from_numpy(q).half().to('cuda')
        k = torch.from_numpy(k).half().to('cuda')
        v = torch.from_numpy(v).half().to('cuda')
        def flash_128(q,k,v,ty):
            o = lsh.hash_triton(q,k,v,ty)
        time_in_microseconds = benchmark_torch_function_in_microseconds(flash_128, q, k, v,'4_128')
        y_128_1_4.append(time_in_microseconds)
    return y_64_1_4,y_128_1_4

if __name__ == "__main__":
    x = np.arange(6)  # 横坐标：0, 1, 2, 3, 4, 5
    y1,y3,y5 = Flash()
    y2,y4,y6 = Ours_sapmele2()
    y4_add,y6_add = Ours_sapmele4()
    ax_x = ["2k","4k","8k","10k","20k","40k"]
    # 画布和子图
    fig, axs = plt.subplots(1, 3, figsize=(20, 4.5))  # 1行3列的子图

    # 宽度和偏移量
    width = 0.3

    # 绘制第一个柱状图
    axs[0].bar(x - width/2, y1, width, label='Flash2', edgecolor='black')  # 设置 label
    axs[0].bar(x + width/2, y2, width, label='DistrAttention', edgecolor='black')  # 设置 label
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([f'{ax_x[i]}' for i in x])
    axs[0].set_ylabel('Time (ms)')  # 设置横坐标标签
    axs[0].set_xlabel('Token length (d=32)')  # 设置纵坐标标签
    #[0].text(0.5, -0.2, '(a) d = 32', transform=axs[0].transAxes, fontsize=20, ha='center')
    # 绘制第二个柱状图
    axs[1].bar(x - width, y3, width, label='Flash2', edgecolor='black')  # 设置 label
    axs[1].bar(x, y4, width, label='DistrAttention (Sampling Rate=2)', edgecolor='black')  # 设置 label
    axs[1].bar(x + width, y4_add, width, label='DistrAttention (Sampling rate=4)', edgecolor='black')  # 设置 label
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f'{ax_x[i]}' for i in x])
    axs[1].set_ylabel('Time (ms)')  # 设置横坐标标签
    axs[1].set_xlabel('Token Length (d=64)')  # 设置纵坐标标签
    #axs[1].text(0.5, -0.2, '(b) d = 64', transform=axs[1].transAxes, fontsize=20, ha='center')
    # 绘制第三个柱状图
    axs[2].bar(x - width, y5, width, label='Group 1', edgecolor='black')  # 设置 label
    axs[2].bar(x, y6, width, label='Group 2', edgecolor='black')  # 设置 label
    axs[2].bar(x + width, y6_add, width, label='Group 3', edgecolor='black')  # 设置 label
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([f'{ax_x[i]}' for i in x])
    axs[2].set_ylabel('Time (ms)')  # 设置横坐标标签
    axs[2].set_xlabel('Token Length (d=128)')  # 设置纵坐标标签
    #axs[2].text(0.5, -0.2, '(c) d = 128', transform=axs[2].transAxes, fontsize=20, ha='center')
    # 共用图例
    handles, labels = axs[1].get_legend_handles_labels()  # 获取第一个子图的图例
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3,frameon=False).get_frame().set_edgecolor('black')   # 调整图例位置

    # 显示图形
    plt.tight_layout()
    plt.show()