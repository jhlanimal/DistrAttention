import torch
import numpy as np
from angular_lsh_tirton1 import AngularLSHTriton
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
np.random.seed(42)
# 创建 4 个示例矩阵
def draw_fig7():
    # 将矩阵放入列表中
    source = np.abs(np.loadtxt('E:/我的论文/atc/simlit/sample_length_source_16.txt'))
    chunk_1 =  np.abs(np.loadtxt('./sample_length_erro_1_2.txt'))
    chunk_2 =  np.abs(np.loadtxt('./sample_length_erro_2_2.txt'))
    chunk_4 =  np.abs(np.loadtxt('./sample_length_erro_4_2.txt'))
    chunk_8 =  np.abs(np.loadtxt('./sample_length_erro_8_2.txt'))

    source_2 =  np.abs(np.loadtxt('./sample_length_source_16.txt'))
    sapmle_len_2 =  np.abs(np.loadtxt('./sample_length_erro_2_2.txt'))
    sapmle_len_4 =  np.abs(np.loadtxt('./sample_length_erro_2_4.txt'))
    sapmle_len_8 =  np.abs(np.loadtxt('./sample_length_erro_2_8.txt'))
    sapmle_len_16 =  np.abs(np.loadtxt('./sample_length_erro_2_16.txt'))
    matrices = [source,chunk_1,chunk_2,chunk_4,chunk_8,source_2,sapmle_len_2,sapmle_len_4,sapmle_len_8,sapmle_len_16]
    #matrices = [source_2,sapmle_len_2,sapmle_len_4,sapmle_len_8,sapmle_len_16]
    #titles = ["chunk = 0", "chunk = 1", "chunk = 2", "chunk = 3"]
    blocks = [0,1,2,4,8,0,2,4,8,16]
    # 创建画布和子图（1 行 4 列）
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))  # 1 行 4 列的画布
    fig.delaxes(axes[1, 0])
    #axes[0, 0].change_geometry(2, 1, 1)
    #fig.suptitle("Heatmaps with Custom Color Range (#E6C5B2 to #7B4B3A)", fontsize=16)
    font = FontProperties(family='Times New Roman', size=10)
    # 找到所有矩阵的最小值和最大值，用于统一颜色坐标尺
    vmin = min(matrix.min() for matrix in matrices)
    vmax = max(matrix.max() for matrix in matrices)

    # 自定义颜色映射：从 #E6C5B2 到 #7B4B3A
    colors = ["#F7C9A5", "#D76B30"]  # 定义颜色范围
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)  # 创建自定义颜色映射

    # 绘制每个矩阵的热力图
    for i, ax in enumerate(axes.flat):
        if i < len(matrices):
            im = ax.imshow(matrices[i], cmap='viridis', vmin=vmin, vmax=vmax)  # 使用自定义颜色映射
            #ax.set_title(titles[i], fontsize=22)
            ax.set_xticks([])  # 隐藏 x 轴刻度
            ax.set_yticks([])  # 隐藏 y 轴刻度
            if i == 0:
                pass
                #ax.text(0.5, -0.1, f"(a) Ground truth", fontproperties=font,transform=ax.transAxes,
                #        fontsize=22, ha='center', va='center')  # 添加标签
                #ax.text(0.5, -0.5, f"Block size=2", fontproperties=font, transform=ax.transAxes,
                #        fontsize=22, ha='center', va='center')  # 添加标签
                #ax.text(0.5, 1.5, f"$G^*$=2", fontproperties=font, transform=ax.transAxes,
                #        fontsize=22, ha='center', va='center')  # 添加标签
            elif i < 5:
                ax.text(0.5, 1.1, f"$l$={blocks[i]}", fontproperties=font, transform=ax.transAxes,
                        fontsize=22, ha='center', va='center')  # 添加标签
            elif i == 5:
                ax.text(0.5, -0.1, f"Approximately", fontproperties=font, transform=ax.transAxes,
                        fontsize=22, ha='center', va='center')  # 添加标签
            else:
                #ax.text(0.5, -0.1, f"({chr(97 + i)}) $G^*$={blocks[i]}", fontproperties=font, transform=ax.transAxes, #加了前面的（a）,(b)序号
                ax.text(0.5, -0.1, f"$G^*$={blocks[i]}", fontproperties=font, transform=ax.transAxes,
                        fontsize=22, ha='center', va='center')  # 添加标签
    #cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical',  fraction=0.0158, pad=0.01)

    ax = axes[0, 0]  # 第一列的子图
    pos = ax.get_position()  # 获取当前子图的位置
    new_pos = [pos.x0, pos.y0 - 0.25, pos.width, pos.height * 1.2]  # 调整位置和高度
    ax.set_position(new_pos)  # 设置新的位置

    for col in range(1, 5):  # 遍历第 2-5 列
        ax1 = axes[0, col]  # 当前列的第一个子图
        ax2 = axes[1, col]  # 当前列的第二个子图

        # 获取当前子图的位置
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()

        # 调整子图的位置
        new_pos1 = [pos1.x0, pos1.y0 - 0.02, pos1.width, pos1.height]  # 上移第一个子图
        new_pos2 = [pos2.x0, pos2.y0 + 0.02, pos2.width, pos2.height]  # 下移第二个子图

        # 设置新的位置
        ax1.set_position(new_pos1)
        ax2.set_position(new_pos2)
        #plt.savefig('simility_chunk_fina_1.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()

def get_error(chunk, sample_len,data_lg,data_key,i_index):
    #chunk = 2
    lsh = AngularLSHTriton(num_projs=16, dim=(1, 1, chunk)).to(device='cuda', dtype=torch.float16)
    q = np.random.rand(100,1,64, 32)
    k = np.random.rand(100,1,64, 32)
    #print(q[0][0])
    torch_q = torch.from_numpy(q).half().to('cuda')
    torch_k = torch.from_numpy(k).half().to('cuda')
    source_attn = torch_q@torch_k.transpose(2,3)
    #print(torch_q[0][0])
    source_attn = source_attn.reshape(100,64,64)
    #print(source_attn[0][0])

    def chunk_size(torch_q,size,sample_len):
        block_num = int(64/size)
        torch_q_tans = torch_q.reshape(100,block_num,size,32)
        hash = lsh.hash_torch(torch_q_tans.permute((0, 1, 3, 2)))
        sorted_values, sorted_indices = torch.sort(hash, dim=2)
        sorted_indices = torch.repeat_interleave(sorted_indices, repeats=chunk, dim=1)
        #q_sample = torch.gather(torch_q.reshape(100,64, 32), dim=2, index=sorted_indices)
        
        A_chunks = torch.split(torch_q.reshape(100, 64, 32), size, dim=1)  # 按行分块
        B_chunks = torch.split(torch_k.reshape(100, 64, 32), size, dim=1)  # 按列分块
        results = []
        count = 0
        for a_chunk in A_chunks:
            index = 0
            result_row = []
            a_chunk_enev = torch.gather(a_chunk, 2, sorted_indices[:,count*size:count*size+size,:])
            x_reshaped = a_chunk_enev.view(100, size, -1, sample_len)  # -1 表示自动计算该维度大小
            a_chunk_sum  = x_reshaped[:,:,:,0].squeeze(-1)
            #print(x_reshaped.shape)
            #a_chunk_sum = torch.sum(x_reshaped, dim=-1)/sample_len #这个求平均值用
            #print(a_chunk_sum.shape)
            #break
            #a_chunk_sum = torch.sum(a_chunk_enev, dim=1)/sample_len#这个用不到
            for b_chunk in B_chunks:
                b_chunk_enev = torch.gather(b_chunk, 2, sorted_indices[:,index*size:index*size+size,:])
                b_reshaped = b_chunk_enev.view(100, size, -1, sample_len)  # -1 表示自动计算该维度大小
                b_chunk_sum = torch.sum(b_reshaped, dim=-1)
                b_chunk = b_chunk_sum.transpose(1, 2)
                result_row.append(a_chunk_sum @ b_chunk)  # 矩阵乘法
                index = index + 1
            results.append(torch.cat(result_row, dim=2))  # 合并列结果
            count = count + 1
        attn = torch.cat(results, dim=1)
        return attn

    approx_attn = chunk_size(torch_q,chunk,sample_len)
    erro = approx_attn# - source_attn
    sum_C = torch.sum(erro, dim=[1, 2])
    mean_C = torch.abs(sum_C/(64 * 64))
    mean_value = torch.mean(mean_C)
    var_value = torch.var(mean_C)
    max_value = torch.max(mean_C)
    min_value = torch.min(mean_C)
    min_index = torch.argmin(mean_C)
    A_14_erro = erro[min_index].cpu().numpy()  # 转换为 NumPy 数组
    A_14_source = source_attn[min_index].cpu().numpy()  # 转换为 NumPy 数组
    data_lg[data_key[i_index]].append(min_value)
    data_lg[data_key[i_index]].append(max_value)
    data_lg[data_key[i_index]].append(mean_value)
        
    #np.savetxt('sample_length_source_16.txt', A_14_source, fmt='%.6f')  # fmt 指定保存的格式
    #np.savetxt('sample_length_erro_'+str(chunk)+'_'+str(sample_len)+'.txt', A_14_erro, fmt='%.6f')  # fmt 指定保存的格式
if __name__ == '__main__':
    l = [1,2,4,8]
    g = [2,4,8,16]
    data_g = {
    'G*=2': [],
    'G*=4': [],
    'G*=8': [],
    'G*=16': []
    }
    data_l = {
    'l=1': [],
    'l=2': [],
    'l=4': [],
    'l=8': []
    }
    data_g_key = ['G=2','G=4','G=8','G=16']
    data_l_key = ['l=1','l=2','l=4','l=8']
    for i in range(len(l)):
        data_l = get_error(l[i], 2, data_l,data_l_key,i)
    for i in range(len(g)):
        data_g = get_error(2, g[i], data_g,data_g_key,i)
    index_labels = ['min', 'max', 'mean']
    df = pd.DataFrame(data_g, index=index_labels)
    print(df)
    df = pd.DataFrame(data_l, index=index_labels)
    print(df)
    draw_fig7()
#print(mean_value,var_value,max_value,min_value)
