"""注释保全检查：确认某个 git 版本里的每一条注释，在当前工作区里都还一字不差地存在。

项目约定是"绝不修改用户写的注释"。改代码时最容易犯的错不是主动改注释，而是重写一段
代码时把嵌在里面的注释连同代码一起换掉、自己毫无察觉。这个脚本就是防这个的
（Plan-1-2-3 那轮靠它抓出过 3 行被静默删掉的注释）。

    tools/check_comments.py [基准版本，默认 HEAD]

判定：先按 token 比对（精确）；token 找不到时退回原文子串比对——注释被整段"注释掉"
保留时（`#     old_code  #原注释`），tokenizer 只会看到最外层那一个 token，
但原文确实还在，这种情况算通过并单独列出来。
"""

import io
import pathlib
import subprocess
import sys
import tokenize


def comments(src: str) -> list[str]:
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True,
                                       text=True).stdout.split() if f.endswith(".py")]
    total = missing = nested = 0
    for f in files:
        old = subprocess.run(["git", "show", f"{base}:{f}"],
                             capture_output=True, text=True).stdout
        if not old:
            continue
        path = pathlib.Path(f)
        new_src = path.read_text() if path.exists() else ""
        new_toks = comments(new_src)
        for c in comments(old):
            total += 1
            if c in new_toks:
                continue
            if c in new_src:            # 被包在更外层的注释里，原文仍在
                nested += 1
                print(f"  [嵌套保留] {f}: {c[:90]}")
                continue
            missing += 1
            print(f"  [缺失] {f}: {c[:90]}")
    print(f"=== 基准 {base}：原始注释 {total} 行，缺失 {missing} 行，"
          f"嵌套保留 {nested} 行 ===")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
