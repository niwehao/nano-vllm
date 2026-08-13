"""nanodeepep 测试共用的构造与判据，改编自 DeepEP/tests/legacy/test_low_latency.py。"""

import random

import torch
import torch.distributed as dist

from nanodeepep import unpack2


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """照抄 DeepEP/deep_ep/utils/math.py:5-9。"""
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def hash_tensor(t: torch.Tensor) -> int:
    return int(t.contiguous().view(torch.uint8).to(torch.int64).sum().item()) ^ \
           int(t.contiguous().view(torch.uint8)[::7].to(torch.int64).sum().item() << 8)


def make_case(num_tokens, hidden, num_experts, num_topk, rank, seed=0, num_masked=10):
    """构造照抄 test_low_latency.py:56-77：
    x 的前 hidden-128 列是常数 (rank-128)（收到后一眼能看出来自哪个 rank），
    最后 128 列写 token 序号（收到后能验证 src_info 对不对），
    随机 scores→topk，再随机戳若干个 -1（不选任何专家）。
    """
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    rank_offset = 128
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * (rank - rank_offset)
    if num_tokens:
        x[:, -128:] = torch.arange(num_tokens, device="cuda").to(torch.bfloat16).view(-1, 1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1].to(torch.int64)
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()
    for _ in range(num_masked if num_tokens else 0):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1
    return x, topk_idx, topk_weights


def check_all(buf, x, topk_idx, topk_weights, group, rank, num_ranks, verbose=True):
    """M2 验收 1/2/3/5：recv_count、数据搬运、加权恒等式、-1 与空手。
    返回 (packed_hash, combined_hash) 供确定性检查复用。"""
    T, K = topk_idx.shape
    L, H = buf.L, buf.H
    rank_offset = 128

    # 全局 topk_idx，用来算"每个专家应该收到多少"
    all_topk_idx = torch.empty((num_ranks, T, K), dtype=topk_idx.dtype, device="cuda")
    dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

    packed, recv_count, handle = buf.low_latency_dispatch(x, topk_idx)

    ph = 0
    for i in range(L):
        expert_id = rank * L + i
        n_valid = int(recv_count[i].item())

        # 判据 1: recv_count == 全局选中该专家的次数
        want = int((all_topk_idx == expert_id).sum().item())
        assert n_valid == want, f"[recv_count] expert {expert_id}: {n_valid} != {want}"

        # layout_range 的 count 之和也要等于它
        lr = handle.layout_range[i].tolist()
        assert sum(unpack2(v)[0] for v in lr) == n_valid, f"[layout] {lr} vs {n_valid}"

        if n_valid == 0:
            continue
        recv_x = packed[i, :n_valid]
        # 判据 2: 前 H-128 列是常数（同一行来自同一个 rank），且等于 src_rank-128
        amin, amax = recv_x[:, :-128].amin(dim=-1), recv_x[:, :-128].amax(dim=-1)
        assert torch.equal(amin, amax), f"[data] expert {expert_id} 行内不是常数"
        # 判据 2': 最后 128 列 == src_info（源 token 下标）
        si = handle.src_info[i, :n_valid]
        assert (recv_x[:, -128:] - si.view(-1, 1)).abs().sum().item() == 0, \
            f"[src_info] expert {expert_id} 的槽位元信息与数据不符"
        # 判据 2'': layout_range 指的那段确实来自那个 rank
        for r in range(num_ranks):
            cnt, beg = unpack2(int(handle.layout_range[i, r].item()))
            assert (amin[beg:beg + cnt] == r - rank_offset).all().item(), \
                f"[layout] expert {expert_id} 段 r={r} 的来源不对"
            assert cnt == int((all_topk_idx[r] == expert_id).sum().item()), \
                f"[layout] expert {expert_id} 段 r={r} 计数不对"
        ph ^= hash_tensor(recv_x)

    # 判据 3: 加权恒等式。假 GEMM = 恒等（专家原样把收到的 token 送回）
    combined = buf.low_latency_combine(packed.clone(), topk_idx, topk_weights, handle)
    ref = x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
    assert torch.isnan(combined).sum().item() == 0, "[combine] 出现 NaN"
    diff = calc_diff(ref, combined) if T else 0.0
    assert diff < 1e-5, f"[combine] 恒等式 diff={diff:.3e} >= 1e-5"

    # 判据 5: 全 -1 的 token 输出必须是 0
    dead = (topk_idx < 0).all(dim=1)
    if dead.any():
        assert combined[dead].abs().sum().item() == 0, "[combine] 全 -1 的行不是 0"

    if verbose and rank == 0:
        print(f"    T={T:>4} 恒等式 diff={diff:.3e}  全 -1 行数={int(dead.sum())}")
    return ph, hash_tensor(combined)
