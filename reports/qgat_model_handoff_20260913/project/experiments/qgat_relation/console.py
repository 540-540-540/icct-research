"""Manual paired-training launcher and read-only live progress/ETA display."""
import argparse
from collections import deque
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/qgat_relation/seed2026'
RESULTS = ROOT / 'results/qgat_relation/seed2026'
CELLS = {'original_D': '原 D', 'relation_D': '新 D（关系旋转）'}
RUNNER = Path(__file__).with_name('run.py')
TITLE = 'QGAT 原 D / 新 D · 40 轮配对训练 · seed 2026'
EPOCHS, ORIGINS, BATCH = 40, 5549, 16
BATCHES = math.ceil(ORIGINS / BATCH)
TOTAL_STEPS = EPOCHS * BATCHES
CHINA = ZoneInfo('Asia/Shanghai')


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def process_alive(pid):
    if not pid:
        return False
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except FileNotFoundError:
        return False
    return str(RUNNER).encode() in command and b'--run' in command


def estimate(progress, live_step_seconds=None, validation_elapsed=0.):
    """Use active compute history; a live wall rate also includes checkpoint I/O."""
    epoch = int(progress.get('epoch', 0))
    steps = int(progress.get('optimizer_steps', 0))
    if epoch >= EPOCHS:
        return dict(total=0., train=0., validation=0., basis='已完成')
    times = progress.get('step_seconds', [])
    first_epoch = 40 if EPOCHS == 100 else 0
    # A tail batch has 13 rather than 16 origins; omit it when estimating full steps.
    full = [seconds for i, seconds in enumerate(times) if i >= first_epoch*BATCHES and (i+1) % BATCHES]
    step_time = live_step_seconds or (statistics.median(full[-64:]) if full else None)
    train = max(0, TOTAL_STEPS-steps) * step_time if step_time is not None else None
    history = progress.get('history', [])
    validation_times = []
    for i, row in enumerate(history):
        if i < first_epoch:
            continue
        before = history[i-1]['elapsed_seconds'] if i else 0.
        computation = sum(times[i*BATCHES:(i+1)*BATCHES])
        validation_times.append(max(0., row['elapsed_seconds']-before-computation))
    validation = statistics.median(validation_times[-3:]) if validation_times else None
    total = None if train is None or validation is None else train + (
        (EPOCHS-epoch)*validation-min(max(0., validation_elapsed), validation))
    basis = '近期实际速度，含写盘' if live_step_seconds else '已有批次耗时参考'
    if EPOCHS == 100 and step_time is None:
        basis = '等待续训阶段新耗时'
    return dict(total=total, train=train, validation=validation, basis=basis)


class CellView:
    def __init__(self, cell):
        self.cell = cell
        self.stamp = None
        self.progress = {}
        self.points = deque()
        self.live_rate = None
        self.pid = None
        self.validation_started = None

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        job = read_json(REPORT / 'jobs' / f'{self.cell}.json')
        running = process_alive(job.get('pid'))
        path = RESULTS / self.cell / 'latest.pt'
        if path.exists():
            stamp = path.stat().st_mtime_ns
            if stamp != self.stamp:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                config = checkpoint['config']
                if config['epochs'] != EPOCHS or config['batch_size'] != BATCH or len(config['indices']) != ORIGINS:
                    raise ValueError(f'{self.cell}: checkpoint does not match the {EPOCHS}-epoch display contract')
                self.progress = checkpoint['progress']
                self.stamp = stamp
        p = self.progress
        steps = p.get('optimizer_steps', 0)
        epoch = p.get('epoch', 0)
        validating = running and epoch < EPOCHS and p.get('cursor') == ORIGINS
        if self.pid != job.get('pid') or not running:
            self.points.clear()
            self.live_rate = None
            self.pid = job.get('pid')
            self.validation_started = None
        if validating:
            # The validation interval is estimated separately, never folded into a train step.
            self.points.clear()
            if self.validation_started is None:
                self.validation_started = now
        elif running:
            self.validation_started = None
            self.points.append((now, steps))
            while len(self.points) > 2 and self.points[1][0] < now-120:
                self.points.popleft()
            start, baseline = self.points[0]
            delta = steps-baseline
            if now-start >= 15 and delta >= 10:
                self.live_rate = (now-start)/delta
        spent_validating = now-self.validation_started if validating else 0.
        eta = estimate(p, self.live_rate, spent_validating)
        complete = epoch >= EPOCHS
        if complete:
            status = '已完成'
        elif validating:
            status = '正在验证'
        elif running:
            status = '训练中' if p else '初始化中'
        else:
            waiting = '等待从40轮续训' if EPOCHS == 100 and not p else '等待启动'
            status = {'interrupted': '已暂停', 'failed': '运行异常',
                      'training': '进程已停止', 'initializing': '进程已停止'}.get(job.get('status'), waiting)
        return dict(cell=self.cell, job=job, progress=p, eta=eta, running=running,
                    complete=complete, status=status)


def duration(seconds):
    if seconds is None:
        return '估算中'
    value = max(0, math.ceil(seconds))
    hours, rest = divmod(value, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def render(rows, owner_running=False):
    now = datetime.now(CHINA)
    all_complete = all(row['complete'] for row in rows)
    active = any(row['running'] for row in rows) or owner_running
    totals = [row['eta']['total'] for row in rows]
    total = max(totals) if all(value is not None for value in totals) else None
    lines = [TITLE,
             f'北京时间 {now:%Y-%m-%d %H:%M:%S}  |  每 1 秒刷新']
    if all_complete:
        lines.append('整体状态：训练已完成  |  剩余 00:00:00')
    elif not active and EPOCHS == 100 and not any(row['progress'] for row in rows):
        lines.append('整体状态：等待从40轮断点续训；首次启动会建立独立分支。')
    elif not active:
        lines.append('整体状态：训练已停止；下面显示断点及启动后的耗时参考。')
    elif total is not None:
        finish = now + timedelta(seconds=total)
        lines.append(f'整体预计剩余 ≈ {duration(total)}  |  预计结束 {finish:%m-%d %H:%M:%S}（北京时间）')
    else:
        train_times = [row['eta']['train'] for row in rows]
        train_only = max(train_times) if all(value is not None for value in train_times) else None
        validation_wait = '续训首轮验证完成后估算' if EPOCHS == 100 else '首轮验证完成后估算'
        lines.append(f'训练部分预计剩余 ≈ {duration(train_only)}  |  完整结束时间：{validation_wait}')
    for row in rows:
        p, eta = row['progress'], row['eta']
        steps = p.get('optimizer_steps', 0)
        ratio = min(1., steps/TOTAL_STEPS)
        filled = int(ratio*24)
        bar = '='*filled + '-'*(24-filled)
        epoch = min(EPOCHS, p.get('epoch', 0)+1)
        cursor = p.get('cursor', 0)
        status = '准备启动' if owner_running and not row['running'] and not row['complete'] else row['status']
        device = row['job'].get('device', {'gnn': 'cuda:0', 'relation_D': 'cuda:1'}.get(row['cell'], '—') if EPOCHS == 100 else '—')
        lines += ['', f'{CELLS[row["cell"]]}  |  {device}  |  {status}']
        if EPOCHS == 100 and not p:
            lines.append('  等待载入40轮父断点；目标共100轮，续训60轮。')
            continue
        round_text = f'第 {epoch}/{EPOCHS} 轮'
        if EPOCHS == 100:
            completed = min(EPOCHS, p.get('epoch', 0))
            round_text = f'已完成 {completed}/{EPOCHS} 轮，剩余 {EPOCHS-completed} 轮'
        lines += [f'  [{bar}] {ratio*100:5.1f}%  |  {round_text}  |  本轮 {cursor}/{ORIGINS}',
                  f'  优化步 {steps}/{TOTAL_STEPS}']
        history = p.get('history', [])
        loss = p.get('loss_sum', 0.)/p['seen'] if p.get('seen', 0) else (history[-1]['loss'] if history else None)
        lines[-1] += f'  |  当前轮平均损失 {loss:.6f}' if loss is not None else '  |  当前轮平均损失 —'
        if eta['total'] is None:
            validation_wait = '续训首轮' if EPOCHS == 100 else '首轮'
            lines.append(f'  预计剩余：训练 ≈ {duration(eta["train"])}；验证耗时待{validation_wait}测得')
        else:
            lines.append(f'  预计剩余 ≈ {duration(eta["total"])}  |  {eta["basis"]}')
        if history:
            last, best = history[-1], min(history, key=lambda r: r['J'])
            lines.append(f'  最近验证：第 {last["epoch"]} 轮 ADE {last["ADE"]:.4f} / FDE {last["FDE"]:.4f} / J {last["J"]:.4f}')
            lines.append(f'  最佳 J：第 {best["epoch"]} 轮 ADE {best["ADE"]:.4f} / FDE {best["FDE"]:.4f} / J {best["J"]:.4f}')
        else:
            lines.append('  验证结果：尚未完成第一轮验证')
    lines += ['', '预计时间会随速度和验证耗时更新。' + ('两组并行，整体取较慢一组。' if len(rows)>1 else ''),
              'Ctrl+C：停止本次启动的训练并保留断点。' if owner_running else '只读监视：Ctrl+C 仅关闭面板。']
    return '\n'.join(lines)


def start_training(resume):
    for path in [REPORT / 'coordinator.json', *(REPORT / 'jobs' / f'{cell}.json' for cell in CELLS)]:
        state = read_json(path)
        if process_alive(state.get('pid')):
            raise RuntimeError('训练已在运行。直接运行 console.py 查看，不要重复使用 --start。')
    exists = (REPORT / 'protocol.json').exists()
    if exists and not resume:
        raise RuntimeError('已有本轮断点，请使用 --start --resume。')
    if resume and not exists:
        raise RuntimeError('本轮尚无可恢复的记录；首次启动只使用 --start。')
    REPORT.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(RUNNER), '--run']
    if resume:
        cmd.append('--resume')
    with (REPORT / 'console_launch.log').open('a', buffering=1) as log:
        return subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))


def main():
    global CELLS, REPORT, RESULTS, RUNNER, TITLE, EPOCHS, BATCHES, TOTAL_STEPS
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true', help='manually start both models and display live progress')
    mode.add_argument('--once', action='store_true', help='one read-only snapshot; never starts training')
    p.add_argument('--resume', action='store_true', help='with --start, continue the saved run in the selected mode')
    models = p.add_mutually_exclusive_group()
    models.add_argument('--gnn', action='store_true', help='historical GNN run with one sample per GPU')
    models.add_argument('--gnn-fast', action='store_true', help='fresh GNN rerun: eight samples per GPU, global batch 16')
    models.add_argument('--extend100', action='store_true', help='GNN and relation QGAT on one GPU each; extend 40 completed epochs to 100')
    args = p.parse_args()
    if args.resume and not args.start:
        p.error('--resume requires --start')
    CELLS = {'original_D': '原 D', 'relation_D': '新 D（关系旋转）'}
    REPORT = ROOT/'reports/qgat_relation/seed2026'
    RESULTS = ROOT/'results/qgat_relation/seed2026'
    RUNNER = Path(__file__).with_name('run.py')
    TITLE = 'QGAT 原 D / 新 D · 40 轮配对训练 · seed 2026'
    EPOCHS = 100 if args.extend100 else 40
    BATCHES = math.ceil(ORIGINS/BATCH)
    TOTAL_STEPS = EPOCHS*BATCHES
    if args.extend100:
        CELLS = {'gnn': 'GNN', 'relation_D': '新 QGAT（关系旋转）'}
        REPORT = ROOT/'reports/qgat_extend100/seed2026'
        RESULTS = ROOT/'results/qgat_extend100/seed2026'
        RUNNER = Path(__file__).with_name('run_extend100.py')
        TITLE = '两卡各一模型 · 续训到100轮 · seed 2026 · 总批次 16'
    elif args.gnn or args.gnn_fast:
        CELLS = {'gnn': 'GNN baseline'}
        REPORT = ROOT/'reports/qgat_relation_gnn/seed2026'
        RESULTS = ROOT/'results/qgat_relation_gnn/seed2026'
        RUNNER = Path(__file__).with_name('run_gnn.py')
        TITLE = 'GNN baseline · 两卡共同训练一个模型 · seed 2026 · 总批次 16'
        if args.gnn_fast:
            REPORT = ROOT/'reports/qgat_relation_gnn_fast/seed2026'
            RESULTS = ROOT/'results/qgat_relation_gnn_fast/seed2026'
            RUNNER = Path(__file__).with_name('run_gnn_fast.py')
            TITLE += ' · 每卡 8 · 从头训练 40 轮'
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    child = None
    views = [CellView(cell) for cell in CELLS]
    tty = sys.stdout.isatty()
    try:
        if args.start:
            child = start_training(args.resume)
        while True:
            owner_running = child is not None and child.poll() is None
            rows = [view.snapshot() for view in views]
            print(('\033[H\033[J' if tty else '') + render(rows, owner_running), flush=True)
            if args.once or (child is None and all(row['complete'] for row in rows)):
                break
            if child is not None and child.poll() is not None:
                if child.returncode:
                    print('\n启动或训练已停止，最近记录：', flush=True)
                    print('\n'.join((REPORT / 'console_launch.log').read_text().splitlines()[-12:]), flush=True)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n正在停止训练并保留最近断点…' if child and child.poll() is None else '\n已关闭监视面板。', flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
            print('训练已停止。下次保留当前模式，使用 --start --resume 继续。', flush=True)


if __name__ == '__main__':
    main()
