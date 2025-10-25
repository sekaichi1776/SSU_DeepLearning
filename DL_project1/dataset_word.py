# coding: utf-8
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
from np import np  # ✅ 핵심: numpy/cupy 자동 전환

# CuPy 배열을 NumPy로 바꾸는 유틸(시각화/ PIL 변환 시 필요)
def to_cpu(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a

try:
    from PIL import Image
    _USE_PIL = True
except Exception:
    from matplotlib.image import imread, imsave  # noqa: F401
    _USE_PIL = False

def _read_image(path: Path):
    # PIL로 읽으면 우선 CPU배열 -> 이후 np.array(...)로 백엔드 배열화
    if _USE_PIL:
        img_cpu = np.asarray(to_cpu(Image.open(path).convert("L")))  # CPU에서 안전히 배열화
        return np.array(img_cpu)  # ✅ GPU면 여기서 cupy로 올라감
    from matplotlib.image import imread
    arr_cpu = imread(path)
    if arr_cpu.ndim == 3:
        arr_cpu = arr_cpu.mean(axis=2)
    return np.array(arr_cpu)      # ✅ GPU면 cupy로 올라감

def _resize_28x28(img):
    if _USE_PIL:
        # PIL은 CPU 배열만 받음
        im = Image.fromarray(to_cpu(img))
        im = im.resize((28, 28))
        return np.array(np.asarray(im))  # CPU->백엔드
    # PIL 없는 경우: 백엔드 상에서 최근접 보간
    H, W = img.shape[:2]
    xv = np.linspace(0, W - 1, 28)
    yv = np.linspace(0, H - 1, 28)
    XI, YI = np.meshgrid(xv, yv)
    xi = np.clip(np.round(XI).astype(int), 0, W - 1)
    yi = np.clip(np.round(YI).astype(int), 0, H - 1)
    return img[yi, xi]

def _xyxy(xs: List[int], ys: List[int]):
    return min(xs), min(ys), max(xs), max(ys)

def _safe_crop(img, x1, y1, x2, y2, pad=1):
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 + 1: x2 = min(W, x1 + 2)
    if y2 <= y1 + 1: y2 = min(H, y1 + 2)
    return img[y1:y2, x1:x2]

def _to_float_norm(img):
    x = img.astype(np.float32) / 255.0
    m = x.mean(); s = x.std()
    if s < 1e-6: s = 1.0
    return (x - m) / s

def scan_pairs(split_dir: Path):
    pairs = []
    for js in sorted(split_dir.glob("*.json")):
        png = js.with_suffix(".png")
        if png.exists():
            pairs.append((png, js))
    return pairs

def build_vocab(train_dir: Path, vocab_path: Path) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for _, js in scan_pairs(train_dir):
        meta = json.loads(js.read_text(encoding="utf-8"))
        for b in meta.get("bbox", []):
            w = b["data"]
            if w not in vocab:
                vocab[w] = len(vocab)
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab

def one_hot(idx: int, num_classes: int):
    y = np.zeros(num_classes, dtype=np.float32)
    y[idx] = 1.0
    return y

def make_batch(pairs: List[Tuple[Path, Path]], vocab: Dict[str, int], page_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ts = [], []
    # ✅ page_indices 배열을 직접 순회
    for pi in page_indices:
        pi = int(pi)  # ndarray 요소 방지
        png, js = pairs[pi]
        meta = json.loads(js.read_text(encoding="utf-8"))
        page = _read_image(png)
        for b in meta.get("bbox", []):
            if b["data"] not in vocab:
                continue
            x1, y1, x2, y2 = _xyxy(b["x"], b["y"])
            crop = _safe_crop(page, x1, y1, x2, y2, pad=2)
            crop = _resize_28x28(crop)
            Xs.append(crop.astype(np.float32) / 255.0)
            Ts.append(vocab[b["data"]])
    if not Xs:
        return np.zeros((0, 1, 28, 28), np.float32), np.zeros((0,), np.int64)
    X = np.stack(Xs, axis=0)[:, None, :, :]  # (N, 1, 28, 28)
    T = np.array(Ts, dtype=np.int64)
    return X, T
