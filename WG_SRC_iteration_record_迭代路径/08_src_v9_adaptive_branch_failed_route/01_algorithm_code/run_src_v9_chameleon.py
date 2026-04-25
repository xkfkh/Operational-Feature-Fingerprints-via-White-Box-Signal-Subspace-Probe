import subprocess
import sys
from pathlib import Path
import argparse
import os


def auto_find_entry(src_v9: Path) -> Path:
    # 先找最像这次新算法的文件名
    patterns = [
        '*adaptive*branch*.py',
        '*algo1*adaptive*.py',
        '*branch*pca*.py',
        '*.py',
    ]
    seen = []
    for pat in patterns:
        for p in sorted(src_v9.glob(pat)):
            if p not in seen:
                seen.append(p)
    if not seen:
        raise FileNotFoundError(f'在 {src_v9} 里没有找到 .py 文件')
    return seen[0]


def main():
    ap = argparse.ArgumentParser(description='启动 src_v9 的 Chameleon 算法脚本')
    ap.add_argument('--src_v9', type=str,
                    default=r'D:\桌面\MSR实验复现与创新\experiments_g1\\wgsrc_development_workspace\src_v9')
    ap.add_argument('--entry', type=str, default='',
                    help='src_v9 里的入口脚本文件名；不填则自动寻找最像 adaptive-branch 的 .py')
    ap.add_argument('--log_dir', type=str,
                    default=r'D:\桌面\MSR实验复现与创新\experiments_g1\\wgsrc_development_workspace\scripts\results_src_v9_chameleon')
    args = ap.parse_args()

    src_v9 = Path(args.src_v9)
    if not src_v9.exists():
        raise FileNotFoundError(f'找不到 src_v9 路径：{src_v9}')

    if args.entry.strip():
        entry = src_v9 / args.entry.strip()
        if not entry.exists():
            raise FileNotFoundError(f'找不到入口脚本：{entry}')
    else:
        entry = auto_find_entry(src_v9)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / 'src_v9_chameleon_stdout.txt'
    stderr_log = log_dir / 'src_v9_chameleon_stderr.txt'

    print('src_v9 =', src_v9)
    print('entry  =', entry)
    print('stdout =', stdout_log)
    print('stderr =', stderr_log)

    env = os.environ.copy()
    cmd = [sys.executable, str(entry)]

    with open(stdout_log, 'w', encoding='utf-8') as fout, open(stderr_log, 'w', encoding='utf-8') as ferr:
        proc = subprocess.run(cmd, cwd=str(src_v9), stdout=fout, stderr=ferr, text=True)

    print('\n运行结束，返回码 =', proc.returncode)
    if proc.returncode != 0:
        print('程序报错了。先打开这两个文件看：')
        print(stdout_log)
        print(stderr_log)
        raise SystemExit(proc.returncode)

    print('程序成功跑完。输出日志在：')
    print(stdout_log)
    print(stderr_log)


if __name__ == '__main__':
    main()


