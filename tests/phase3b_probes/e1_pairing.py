"""E1:本机有没有可配对的目标/草稿模型。

配对的硬条件是**同一个 tokenizer + 同一个词表**:接受判定比的是 token id,
两边 id 对不上,"草稿 token d 是不是等于目标 argmax"这句话就没有意义。

这里不看"名字像不像同一家族",只做三件可证伪的检查:
  1. config.json 的 vocab_size 严格相等(vLLM 也只查这一条,见报告 vLLM 章节);
  2. tokenizer.json / vocab.json / merges.txt 逐字节相同(比"能编码出同样结果"更强);
  3. 拿一段真实语料 round-trip 编码,两边 id 序列逐元素相同(前两条的独立复核)。
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

HF = os.path.expanduser("~/huggingface")


def sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def inventory():
    rows = []
    for name in sorted(os.listdir(HF)):
        d = os.path.join(HF, name)
        cfg_path = os.path.join(d, "config.json")
        if not os.path.isdir(d) or not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
                   if f.endswith(".safetensors"))
        rows.append(dict(
            name=name, path=d, arch=cfg["architectures"][0],
            vocab=cfg.get("vocab_size"), layers=cfg.get("num_hidden_layers"),
            hidden=cfg.get("hidden_size"), kv_heads=cfg.get("num_key_value_heads"),
            head_dim=cfg.get("head_dim"), bytes=size,
            tok_sha=sha(os.path.join(d, "tokenizer.json")),
        ))
    return rows


def main():
    rows = inventory()
    print("=" * 100)
    print("E1-1 本机模型清单")
    print("=" * 100)
    print(f"{'目录':<18}{'架构':<22}{'词表':>8}{'层':>5}{'hidden':>8}"
          f"{'kv头':>6}{'head_dim':>9}{'权重GiB':>10}  tokenizer.json sha")
    for r in rows:
        print(f"{r['name']:<18}{r['arch']:<22}{r['vocab']:>8}{r['layers']:>5}"
              f"{r['hidden']:>8}{r['kv_heads']:>6}{r['head_dim']:>9}"
              f"{r['bytes']/2**30:>10.2f}  {r['tok_sha']}")

    by = {r["name"]: r for r in rows}
    tgt, drf = by.get("Qwen3-8B"), by.get("Qwen3-0.6B")
    print()
    print("=" * 100)
    print("E1-2 候选配对:目标 Qwen3-8B × 草稿 Qwen3-0.6B")
    print("=" * 100)
    if tgt is None or drf is None:
        print("  ✗ 缺少其中一个,无法配对")
        return 1

    ok = True
    same_vocab = tgt["vocab"] == drf["vocab"]
    ok &= same_vocab
    print(f"  [{'✓' if same_vocab else '✗'}] vocab_size 相等: "
          f"{tgt['vocab']} vs {drf['vocab']}")

    for fn in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        a, b = sha(os.path.join(tgt["path"], fn)), sha(os.path.join(drf["path"], fn))
        same = a is not None and a == b
        # tokenizer_config.json 允许不同(里面有 chat template、model_max_length 之类
        # 与 id 映射无关的字段),只报不判
        hard = fn != "tokenizer_config.json"
        if hard:
            ok &= same
        print(f"  [{'✓' if same else ('✗' if hard else '·')}] {fn:<22} "
              f"{a} vs {b}" + ("" if hard else "   (软条件,不影响 id 映射)"))

    from transformers import AutoTokenizer
    from common import CORPUS
    t1 = AutoTokenizer.from_pretrained(tgt["path"], use_fast=True)
    t2 = AutoTokenizer.from_pretrained(drf["path"], use_fast=True)
    i1, i2 = t1.encode(CORPUS), t2.encode(CORPUS)
    same_ids = i1 == i2
    ok &= same_ids
    print(f"  [{'✓' if same_ids else '✗'}] 真实语料 round-trip: {len(i1)} vs {len(i2)} "
          f"个 token,{'逐元素相同' if same_ids else '不同'}")
    print(f"  [{'✓' if t1.eos_token_id == t2.eos_token_id else '✗'}] eos_token_id: "
          f"{t1.eos_token_id} vs {t2.eos_token_id}")

    print()
    print(f"E1 结论:{'配对可用' if ok else '配对不可用'}")
    print(f"  目标 Qwen3-8B  权重 {tgt['bytes']/2**30:.2f} GiB "
          f"({tgt['layers']} 层, hidden {tgt['hidden']})")
    print(f"  草稿 Qwen3-0.6B 权重 {drf['bytes']/2**30:.2f} GiB "
          f"({drf['layers']} 层, hidden {drf['hidden']})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
