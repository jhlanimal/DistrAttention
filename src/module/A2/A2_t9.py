import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time


import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.cuda import Stream

from angular import AngularLSHTriton

import torch.utils.benchmark as benchmark

from torch.backends.cuda import sdp_kernel, SDPBackend
import os
import pandas as pd

def read_data():#读取数据
    data = {
    'gpu=1': [],
    'gpu=2': [],
    'gpu=4': []
    }
    with open('timing_results_distr_1_0.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=1'].append(values[0])
    with open('timing_results_distr_2_0.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=2'].append(values[0])
    with open('timing_results_distr_4_0.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=4'].append(values[0])
    with open('timing_results_distr_1_1.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=1'].append(values[0])
    with open('timing_results_distr_2_1.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=2'].append(values[0])
    with open('timing_results_distr_4_1.txt', 'r') as file:
        line = file.readline().strip()  # 读取第一行并去除首尾空白字符
        values = line.split(',')
        data['gpu=4'].append(values[0])
    index_labels = ['Flash', 'Ours']
    df = pd.DataFrame(data, index=index_labels)
    print(df)
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

lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, 128)).to(device='cuda', dtype=torch.float16)

def triple_matmul_3d_pipeline(rank, q, k, v, world_size, D, M, N, P, Q, micro_chunk,is_ours):
    # 初始化时间和事件记录（仅Rank 0记录）
    if rank == 0:
        timers = {
            "total_start": torch.cuda.Event(enable_timing=True),
            "total_end": torch.cuda.Event(enable_timing=True),
            "scatter_events": [],
            "compute_events": [],
            "gather_events": []
        }
        timers["total_start"].record()
    else:
        timers = None

    # 原初始化代码保持不变
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        full_D = torch.empty(D, M, N, device='cuda')
    else:
        full_D = None
    
    d_per_gpu = D // world_size
    device = torch.device(f'cuda:{rank}')
    
    A_local = torch.empty(micro_chunk, M, N, device=device).half()
    B_local = torch.empty(micro_chunk, M, N, device=device).half()
    C_local = torch.empty(micro_chunk, M, N, device=device).half()
    
    A_buffers = [
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half(),
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half()
    ]
    B_buffers = [
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half(),
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half()
    ]
    C_buffers = [
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half(),
        torch.empty((world_size*micro_chunk, M, N), device='cuda').half()
    ]
    
    gather_list = None
    if rank == 0:
        gather_list = [torch.empty(micro_chunk, M, N, device='cuda').half() for _ in range(world_size)]
        full_D = torch.empty(D, M, N, device='cuda')
    else:
        full_D = None
    
    compute_stream = torch.cuda.current_stream()
    gather_stream = torch.cuda.Stream()
    comm_stream_a = torch.cuda.Stream()
    comm_stream_b = torch.cuda.Stream()
    comm_stream_c = torch.cuda.Stream()
    
    current_a = 0
    current_b = 0
    current_c = 0
    
    if rank == 0:
        scatter_A = list(torch.split(q, micro_chunk, dim=0))
        scatter_B = list(torch.split(k, micro_chunk, dim=0))
        scatter_C = list(torch.split(v, micro_chunk, dim=0))
    else:
        scatter_A, scatter_B, scatter_C = None, None, None

    # 添加Scatter阶段计时
    scatter_events = []
    for stream in [comm_stream_a, comm_stream_b, comm_stream_c]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        scatter_events.append((start_event, end_event))
    
    # 原Scatter代码保持不变
    with torch.cuda.stream(comm_stream_a):
        if rank == 0:
            A_buffers[current_a].copy_(torch.cat(scatter_A[0:world_size]))
            scatter_list_a = [A_buffers[current_a][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
        else:
            scatter_list_a = None
        dist.scatter(A_local, scatter_list_a, src=0)
    with torch.cuda.stream(comm_stream_b):
        if rank == 0:
            B_buffers[current_b].copy_(torch.cat(scatter_B[0:world_size]))
            scatter_list_b = [B_buffers[current_b][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
        else:
            scatter_list_b = None
        dist.scatter(B_local, scatter_list_b, src=0)
    with torch.cuda.stream(comm_stream_c):
        if rank == 0:
            C_buffers[current_c].copy_(torch.cat(scatter_C[0:world_size]))
            scatter_list_c = [C_buffers[current_c][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
        else:
            scatter_list_c = None
        dist.scatter(C_local, scatter_list_c, src=0)
    
    # 记录Scatter结束时间
    for idx, stream in enumerate([comm_stream_a, comm_stream_b, comm_stream_c]):
        scatter_events[idx][1].record(stream)

    K = d_per_gpu//micro_chunk
    for block_idx in range(K):
        next_a = 1 - current_a
        next_b = 1 - current_b
        next_c = 1 - current_c
        
        if block_idx < K-1:
            # 预加载计时
            pre_scatter_events = []
            for stream in [comm_stream_a, comm_stream_b, comm_stream_c]:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                pre_scatter_events.append((start, end))
            
            # 原预加载代码保持不变
            with torch.cuda.stream(comm_stream_b):
                if rank == 0:
                    A_buffers[next_a].copy_(torch.cat(scatter_A[world_size*(block_idx+1):world_size*(block_idx+1)+world_size]))
                    scatter_list_a = [A_buffers[next_a][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
                else:
                    scatter_list_a = None
                dist.scatter(A_local, scatter_list_a, src=0)
            with torch.cuda.stream(comm_stream_b):
                if rank == 0:
                    B_buffers[next_b].copy_(torch.cat(scatter_B[world_size*(block_idx+1):world_size*(block_idx+1)+world_size]))
                    scatter_list_b = [B_buffers[next_b][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
                else:
                    scatter_list_b = None
                dist.scatter(B_local, scatter_list_b, src=0)
            with torch.cuda.stream(comm_stream_c):
                if rank == 0:
                    C_buffers[next_c].copy_(torch.cat(scatter_C[world_size*(block_idx+1):world_size*(block_idx+1)+world_size]))
                    scatter_list_c = [C_buffers[next_c][i*micro_chunk: (i+1)*micro_chunk] for i in range(world_size)]
                else:
                    scatter_list_c = None
                dist.scatter(C_local, scatter_list_c, src=0)
            
            # 记录预加载结束时间
            for idx, stream in enumerate([comm_stream_a, comm_stream_b, comm_stream_c]):
                pre_scatter_events[idx][1].record(stream)
            if rank == 0:
                timers["scatter_events"].extend(pre_scatter_events)

        # 计算阶段计时
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record(compute_stream)
        
        # 原计算代码保持不变
        compute_stream.wait_stream(comm_stream_a)
        compute_stream.wait_stream(comm_stream_b)
        compute_stream.wait_stream(comm_stream_c)
        
        current_A = A_local
        current_B = B_local
        current_C = C_local
        D_block = lsh.hash_triton(current_A.unsqueeze(1),current_B.unsqueeze(1),current_C.unsqueeze(1))
        D_block = D_block.squeeze(1)
        
        compute_end.record(compute_stream)
        if rank == 0:
            timers["compute_events"].append((compute_start, compute_end))

        # 收集阶段计时
        gather_start = torch.cuda.Event(enable_timing=True)
        gather_end = torch.cuda.Event(enable_timing=True)
        gather_start.record(gather_stream)
        
        # 原收集代码保持不变
        current_a = next_a
        current_b = next_b
        current_c = next_c
        with torch.cuda.stream(gather_stream):
            dist.gather(D_block, gather_list if rank == 0 else None, dst=0)
            if rank == 0:
                for i in range(world_size):
                    start_idx = i * d_per_gpu + block_idx * micro_chunk
                    full_D[start_idx:start_idx + micro_chunk] = gather_list[i]
        
        gather_end.record(gather_stream)
        if rank == 0:
            timers["gather_events"].append((gather_start, gather_end))

        torch.cuda.synchronize()
        dist.barrier()

    # 统计总时间
    if rank == 0:
        timers["total_end"].record()
        torch.cuda.synchronize()
        
        # 计算各阶段时间
        def calc_time(event_pairs):
            return sum(s.elapsed_time(e) for s, e in event_pairs)
        
        total_time = timers["total_start"].elapsed_time(timers["total_end"])
        scatter_time = calc_time(timers["scatter_events"])
        compute_time = calc_time(timers["compute_events"])
        gather_time = calc_time(timers["gather_events"])
        
        #print(f"\n=== 耗时分析 ===")
        #print(f"总耗时: {compute_time+gather_time+scatter_time:.2f} ms")
        #print(f"Scatter总耗时: {scatter_time:.2f} ms")
        #print(f"计算总耗时: {compute_time:.2f} ms")
        #print(f"Gather总耗时: {gather_time:.2f} ms")
        #print(f"计算占比: {compute_time/(compute_time+gather_time+scatter_time)*100:.1f}%") 
        result_str = (f"{compute_time+gather_time+scatter_time:.2f},{scatter_time:.2f},{compute_time:.2f},"
                 f"{gather_time:.2f},{compute_time/(compute_time+gather_time+scatter_time)*100:.1f}%")
        with open("timing_results_distr_"+str(world_size)+"_"+str(is_ours)+".txt", "a") as f:
            f.write(result_str + "\n")
    #dist.destroy_process_group()
    return None
    '''
    # 本地计算
    T = torch.bmm(A_local, B_local)  # [d_per_gpu, M, P]
    D_local = torch.bmm(T, C_local)  # [d_per_gpu, M, Q]
    
    # 收集结果到主节点
    gathered_D = [torch.empty_like(D_local) for _ in range(world_size)]
    dist.all_gather(gathered_D, D_local)
    
    if rank == 0:
        final_D = torch.cat(gathered_D, dim=0)  # 拼接为 [D, M, Q]
        print(f"Final Result Shape: {final_D.shape}")
        return final_D
    return None
    '''
if __name__ == "__main__":
    
    D = 480       # 总batch数 480->80
    M, N = 2048, 128
    P, Q = 8000, 5000
    q = np.random.rand(D, M, N)
    k = np.random.rand(D, M, N)
    v = np.random.rand(D, M, N)
    q = torch.from_numpy(q).half().to('cuda')
    k = torch.from_numpy(k).half().to('cuda')
    v = torch.from_numpy(v).half().to('cuda')

    #o = lsh.flashattn_v2_triton(q,k,v)
    #print(o.shape)
    
    #M, N = 1000, 1000
    #P, Q = 8000, 5000
    def distr(world_size,is_ours,micro_chunk=40):#is_ours=1,是我们的方法
        for i in range(1):
            #micro_chunk = 40
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12357'
            #world_size = 2  # 确保 D % world_size == 0
            mp.spawn(triple_matmul_3d_pipeline,
                    args=(q,k,v,world_size, D, M, N, P, Q, micro_chunk,is_ours),
                    nprocs=world_size)
    
    distr(1,1)
    distr(2,1)
    distr(4,1,20)
    distr(1,0)
    distr(2,0)
    distr(4,0,20)
    read_data()
