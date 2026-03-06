"""
hazirla_ve_egit.py - Real-world dataset ile fine-tune egitimi
RTX 3050 6GB + i5-13500H

Kullanim:
  python hazirla_ve_egit.py --resume checkpoints/stage1_full/checkpoint_epoch_14.pth
  python hazirla_ve_egit.py --resume checkpoints/stage1_full/checkpoint_epoch_14.pth --epochs 30 --lr 5e-5
  python hazirla_ve_egit.py --resume checkpoints/realworld_finetune/checkpoint_epoch_20.pth --reset-best-acc
"""
import subprocess, sys, os, time, argparse
from pathlib import Path
from datetime import timedelta

ROOT = Path(__file__).parent

def banner(msg): print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)
def ok(msg):     print(f"  [OK] {msg}", flush=True)

# ── CLI argumanlari ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Real-world fine-tune egitimi')
parser.add_argument('--resume', type=str, required=True,
                    help='Baslangic checkpoint (ornek: checkpoints/stage1_full/checkpoint_epoch_14.pth)')
parser.add_argument('--train_json', type=str,
                    default='data/real_world_train.json')
parser.add_argument('--val_json',   type=str,
                    default='data/real_world_test.json')
parser.add_argument('--save_dir',   type=str,
                    default='checkpoints/realworld_finetune')
parser.add_argument('--epochs',     type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr',         type=float, default=5e-5)
parser.add_argument('--num_workers',type=int, default=4)
parser.add_argument('--no-augment', action='store_true', default=False)
parser.add_argument('--reset-best-acc', action='store_true', default=False,
                    help='best_val_acc esigini sifirla (farkli veriyle devam ederken kullan)')
args = parser.parse_args()

banner("REAL-WORLD FINE-TUNE EGITIMI")

import torch
ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
resume_epoch = ckpt.get('epoch', 0)
resume_acc   = ckpt.get('best_val_acc', 0.0)
del ckpt
ok(f"Resume : {Path(args.resume).name}  (epoch={resume_epoch}, best_acc={resume_acc*100:.2f}%)")
ok(f"Train  : {args.train_json}")
ok(f"Val    : {args.val_json}")
ok(f"Kayit  : {args.save_dir}")
ok(f"Epochs : {resume_epoch + 1} → {resume_epoch + args.epochs}  ({args.epochs} epoch)")
ok(f"LR     : {args.lr}")
ok(f"Augment: {'Kapali' if args.no_augment else 'Aktif'}")
if args.reset_best_acc:
    ok(f"Reset best_acc: Evet  ({resume_acc*100:.2f}% → 0.00%)")

cmd = [
    sys.executable,
    str(ROOT / 'training' / 'train_recognition_mjsynth.py'),
    '--train_json',  str(ROOT / args.train_json),
    '--val_json',    str(ROOT / args.val_json),
    '--resume',      str(args.resume),
    '--save_dir',    str(ROOT / args.save_dir),
    '--epochs',      str(resume_epoch + args.epochs),
    '--batch_size',  str(args.batch_size),
    '--lr',          str(args.lr),
    '--num_workers', str(args.num_workers),
    '--quiet',
]
if not args.no_augment:
    cmd.append('--augment')
if args.reset_best_acc:
    cmd.append('--reset-best-acc')

banner("EGITIM BASLIYOR")
t0 = time.time()
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
    cwd=str(ROOT),
    env={**os.environ, 'PYTHONPATH': str(ROOT)},
)
for line in proc.stdout:
    print(line, end='', flush=True)
proc.wait()

elapsed = time.time() - t0
banner(f"TAMAMLANDI  ({timedelta(seconds=int(elapsed))})")
if proc.returncode != 0:
    ok(f"HATA — cikis kodu: {proc.returncode}")
    sys.exit(proc.returncode)

ok(f"Checkpointler: {ROOT / args.save_dir}")
for f in sorted((ROOT / args.save_dir).glob('*.pth')):
    print(f"  {f.name:45s} {f.stat().st_size/1024**2:.1f} MB")
