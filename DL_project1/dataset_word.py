# coding: utf-8
# =============================================================================
#  dataset_word.py — 페이지 JSON(label) + PNG(image)에서 단어 단위 크롭 배치 생성
#
# ※ 참고 
#   - 이미지 로딩/흑백 변환 아이디어      → ch1/img_show.py, ch3/mnist_show.py
#   - (N, 1, 28, 28) 배치 텐서 형상        → ch3/neuralnet_mnist.py, ch7/simple_convnet.py
#   - 정규화(float32/255.0) 관례           → ch3/sigmoid.py, relu.py, neuralnet_mnist.py 일련의 예제들
#   - 안전한 슬라이싱/방어적 처리 습관     → ch4/two_layer_net.py, ch5/layer_naive.py 계열 전반의 방어 로직 스타일
# =============================================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
from np import np 

#  CuPy → NumPy 변환 유틸 (시각화/PIL 입출력 시 CPU 배열이 필요)
#   - 참고: ch1/img_show.py, ch3/mnist_show.py
def to_cpu(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a

# 이미지 로딩 (PNG → 회색조)
#   - PIL이 있으면 PIL로 로드(흑백 변환) → CPU ndarray → 백엔드(np.array)로 승격
#   - PIL이 없으면 matplotlib.image.imread 사용
#   - 참고: ch1/img_show.py, ch3/mnist_show.py (이미지 흑백/시각화 루틴)
try:
    from PIL import Image
    _USE_PIL = True
except Exception:
    from matplotlib.image import imread, imsave  # noqa: F401
    _USE_PIL = False

def _read_image(path: Path):
    # PIL 경로: PIL은 CPU 배열만 처리 → np.array(...)로 백엔드(cupy/numpy) 배열화
    if _USE_PIL:
        img_cpu = np.asarray(to_cpu(Image.open(path).convert("L"))) 
        return np.array(img_cpu)  # GPU이면 cupy
    # matplotlib 경로: CPU로 읽고 백엔드로
    from matplotlib.image import imread
    arr_cpu = imread(path)
    if arr_cpu.ndim == 3:  # RGB → grayscale (간단 평균)
        arr_cpu = arr_cpu.mean(axis=2)
    return np.array(arr_cpu)

# 리사이즈: 임의 크롭 → 28x28
#   - PIL이 있으면 PIL.resize 사용
#   - 없으면 최근접 보간(정수 인덱싱)으로 간단히 28x28 리샘플
#   - 참고: ch3/neuralnet_mnist.py 
def _resize_28x28(img):
    if _USE_PIL:
        im = Image.fromarray(to_cpu(img))  # PIL은 CPU 배열만 허용
        im = im.resize((28, 28))
        return np.array(np.asarray(im))    # CPU → 백엔드 배열
    # PIL 없는 경우: 백엔드 상에서 최근접 보간 (정수 위치)
    H, W = img.shape[:2]
    xv = np.linspace(0, W - 1, 28)
    yv = np.linspace(0, H - 1, 28)
    XI, YI = np.meshgrid(xv, yv)
    xi = np.clip(np.round(XI).astype(int), 0, W - 1)
    yi = np.clip(np.round(YI).astype(int), 0, H - 1)
    return img[yi, xi]

# bbox 유틸
#   - JSON의 x[], y[](꼭짓점들) → (x1,y1,x2,y2) 박스
def _xyxy(xs: List[int], ys: List[int]):
    return min(xs), min(ys), max(xs), max(ys)

# 안전 크롭: 경계/너비/높이 1픽셀 이하 예외 방지
#   - 참고: ch4~5
def _safe_crop(img, x1, y1, x2, y2, pad=1):
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 + 1: x2 = min(W, x1 + 2)  # 최소 1px 확보
    if y2 <= y1 + 1: y2 = min(H, y1 + 2)
    return img[y1:y2, x1:x2]

# 간단 정규화 (z-score 형태; 현재는 미사용)
#   - 참고: ch3/neuralnet_mnist.py
def _to_float_norm(img):
    x = img.astype(np.float32) / 255.0
    m = x.mean(); s = x.std()
    if s < 1e-6: s = 1.0
    return (x - m) / s

# split 디렉터리에서 (PNG, JSON) 쌍 스캔
#   - 참고: ch3/mnist_show.py가 파일을 순회/읽는 흐름
def scan_pairs(split_dir: Path):
    pairs = []
    for js in sorted(split_dir.glob("*.json")):
        png = js.with_suffix(".png")
        if png.exists():
            pairs.append((png, js))
    return pairs

# vocab 구축: train/*.json의 bbox.data(단어 문자열) → 인덱스
#   - 참고: ch3/neuralnet_mnist.py의 label 인덱싱 관례를 본 프로젝트에 맞게 적용
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

# one-hot 유틸
#   - 참고: ch3~5 전반에서의 one-hot 표현 관례 (실제 학습은 sparse index 사용)
def one_hot(idx: int, num_classes: int):
    y = np.zeros(num_classes, dtype=np.float32)
    y[idx] = 1.0
    return y

# make_batch: 페이지 인덱스들 → (X, T)
#   - 입력:
#       pairs         : [(png_path, json_path), ...]
#       vocab         : { token(str) : id(int) }
#       page_indices  : [int, int, ...]  (셔플된 페이지 인덱스 일부)
#   - 출력:
#       X: (N, 1, 28, 28)  float32  [0..1]  (ch3/ch7 형상 관례)
#       T: (N,)            int64    (클래스 인덱스)
#
#   - 내부:
#       각 페이지의 bbox를 순회 → 안전 크롭 → 28x28 → 누적
#       비어 있으면 안전하게 (0-size) 텐서 반환 (학습 루프 방어)
#
#   - 참고:
#       형상/정규화/배치 관례 → ch3/neuralnet_mnist.py, ch7/simple_convnet.py
#       방어 로직/빈 배치 처리 → ch4/train_neuralnet.py, ch5/train_neuralnet.py
def make_batch(pairs: List[Tuple[Path, Path]], vocab: Dict[str, int], page_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ts = [], []
    # page_indices가 numpy 배열이어도 안전하도록 int(pi) 강제 
    for pi in page_indices:
        pi = int(pi)
        png, js = pairs[pi]
        meta = json.loads(js.read_text(encoding="utf-8"))
        page = _read_image(png)

        for b in meta.get("bbox", []):
            if b["data"] not in vocab:
                continue
            x1, y1, x2, y2 = _xyxy(b["x"], b["y"])
            crop = _safe_crop(page, x1, y1, x2, y2, pad=2)
            crop = _resize_28x28(crop)
            # 간단 스케일링: [0,1] (※ ch3 관례)
            Xs.append(crop.astype(np.float32) / 255.0)
            Ts.append(vocab[b["data"]])

    # 빈 배치 방어: (0,1,28,28), (0,)
    if not Xs:
        return np.zeros((0, 1, 28, 28), np.float32), np.zeros((0,), np.int64)

    # (N, H, W) → (N, 1, 28, 28)
    X = np.stack(Xs, axis=0)[:, None, :, :]
    T = np.array(Ts, dtype=np.int64)
    return X, T
