# coding: utf-8
# ============================================================================
# train_wordnet.py — WordNet 학습 스크립트
#   - Adam 최적화, 배치 페이지 단위 로딩, 한글-only vocab 필터, 로그/파라미터 저장
#   - Optimizer(Adam) 업데이트 수식/패턴  → ch6_1/optimizer_compare_naive.py, optimizer_compare_mnist.py
#   - 학습 루프/로그 저장 패턴           → ch4/train_neuralnet.py, ch5/train_neuralnet.py
#   - Mini-batch 전처리/배치 구성         → ch3/neuralnet_mnist_batch.py (배치 처리 흐름 참고)
#   - Network forward/backward/params     → ch7/simple_convnet.py (Convolution/Pooling/Affine/SoftmaxWithLoss 구조)
#   - 데이터 스캔/로딩/전처리             → 본 프로젝트의 dataset_word.py 
#   - 모델 본체(CNN WordNet)             → wordnet.py 
# ============================================================================

from __future__ import annotations
import json, re, gc
from pathlib import Path
from np import np  # cupy/numpy 자동 전환 래퍼 (np.py)

from dataset_word import scan_pairs, build_vocab, make_batch  # ✅ 너가 준 헬퍼
from wordnet import WordNet                                  # ✅ 너가 준 CNN 모델

# Adam Optimizer
#   - 파라미터별 1차/2차 모멘트 추정(m, v)와 bias correction을 포함한 업데이트
#   - 수식/업데이트 흐름은 ch6_1/optimizer_compare_naive.py, optimizer_compare_mnist.py와 동일한 형태
class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def update(self, params, grads):
        # t(스텝) 증가 및 bias-corrected learning rate
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t))
        for k in params.keys():
            # 모멘트 버퍼 초기화
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            # 1차/2차 모멘트 업데이트  (※ ch6_1/optimizer_compare_naive.py 패턴)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (grads[k] ** 2)
            # 파라미터 갱신
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)

# Main Training Loop
#   - 전체 구조/흐름은 ch4/train_neuralnet.py, ch5/train_neuralnet.py의 학습 루프를 참고하여 구성
#   - 차이점: 단어 단위 OCR을 위한 make_batch, 한글-only vocab 필터, GPU 안전 평균, 메모리 관리 등
def main():
    # 데이터 루트 
    data_root = Path("../DL_project1/data")
    train_dir = data_root / "train"
    test_dir  = data_root / "test"

    # vocab 로딩/생성 
    vocab_path = train_dir / "vocab.json"
    if not vocab_path.exists():
        # train 디렉터리의 json에서 단어 사전 구축  (dataset_word.py: build_vocab)
        vocab = build_vocab(train_dir, vocab_path)
    else:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    # 한글-only 필터링 (숫자/영어/한자 제외)
    vocab = {k: v for k, v in vocab.items() if re.fullmatch(r"[가-힣]+", k)}
    V = len(vocab)
    print("Vocab size:", V, flush=True)

    # 페이지(이미지-라벨) 쌍 목록 스캔 (dataset_word.py: scan_pairs)
    train_pairs = scan_pairs(train_dir)
    test_pairs  = scan_pairs(test_dir)
    print("pages: train", len(train_pairs), "test", len(test_pairs), flush=True)

    # 모델 초기화 (wordnet.py: ch7/simple_convnet.py 구조를 단어 분류로 확장)
    net = WordNet(vocab_size=V, weight_init_std=0.01)

    # Hyperparameters
    #  - Adam 설정은 ch6_1/optimizer_compare_* 코드들과 동일한 기본 값에서 lr만 상향
    optim = Adam(lr=3e-3)
    epochs = 10
    pages_per_epoch     = len(train_pairs)  # 전체 페이지 사용 (초기 min cap 제거)
    batch_pages         = 256               # 페이지 단위 미니배치 (메모리/속도 트레이드오프, 실험적으로 조정)
    MAX_SAMPLES_PER_ITER = 1024             # 한 iter에서 학습할 crop 상한 (OOM 방지)

    # 로그 버퍼 (※ ch4/ch5 train_neuralnet.py에서 로그 저장 패턴 참고)
    train_loss_log, train_acc_log, test_acc_log = [], [], []

    # ___EPOCH LOOP ___
    for ep in range(1, epochs + 1):
        print(f"Epoch {ep}/{epochs}", flush=True)
        idxs = np.arange(len(train_pairs))
        np.random.shuffle(idxs)  # 데이터 섞기 (ch3/neuralnet_mnist_batch.py의 배치 섞기 흐름 참고)
        page_ptr = 0
        losses, train_accs = [], []

        # ____ Iterations (by page batch) ____
        while page_ptr < pages_per_epoch:
            # 미니배치로 사용할 페이지 인덱스 선택
            batch_idxs = idxs[page_ptr: page_ptr + batch_pages]

            # 페이지에서 단어 bbox를 크롭하여 (N,1,28,28), (N,) 로 변환  (dataset_word.py: make_batch)
            X, T = make_batch(train_pairs, vocab, batch_idxs)
            page_ptr += batch_pages

            # 크롭이 없는 페이지 배치 방어
            if X.shape[0] == 0:
                continue

            # 너무 많은 크롭은 샘플링해 상한 제한 (GPU 메모리 안정화)
            if X.shape[0] > MAX_SAMPLES_PER_ITER:
                sel = np.random.choice(X.shape[0], size=MAX_SAMPLES_PER_ITER, replace=False)
                X = X[sel]; T = T[sel]

            # Forward & Loss (ch5/train_neuralnet.py의 손실 계산 흐름과 동일)
            loss = net.loss(X, T)
            losses.append(loss)
            train_loss_log.append(loss)

            # Backward & Update (ch5/train_neuralnet.py → gradient/optimizer 호출 흐름)
            grads = net.gradient(X, T)              # wordnet.py 내부 레이어들의 backward 합성 (ch7/simple_convnet.py 참고)
            optim.update(net.params, grads)         # Adam 업데이트 (ch6_1/optimizer_compare_*.py)

            # Quick train accuracy on current mini-batch
            train_accs.append(net.accuracy(X, T, batch_size=max(1, X.shape[0] // 2)))

            # 진행 상황 출력 (iter 단위)
            if len(losses) % 20 == 0:
                pct = 100.0 * page_ptr / pages_per_epoch
                print(f"   pages {page_ptr:5d}/{pages_per_epoch} ({pct:5.1f}%) "
                      f"| iter {len(losses):05d} | loss {loss:.4f}", flush=True)

            # 메모리 해제 (GPU/CPU 공통) — 대규모 배치 처리 안정화
            del X, T, grads
            gc.collect()

        # ____ Validation (빠른 추정) _____
        # ch4/ch5 train_neuralnet.py의 검증 루프 패턴을 간소화하여 적용
        idxs = np.arange(len(test_pairs))
        np.random.shuffle(idxs)
        test_acc, total = 0.0, 0
        for pi in idxs[:5]:  # 빠른 추정을 위해 일부 페이지만 샘플링
            Xv, Tv = make_batch(test_pairs, vocab, [pi])
            if Xv.shape[0] == 0:
                continue
            if Xv.shape[0] > MAX_SAMPLES_PER_ITER:
                sel = np.random.choice(Xv.shape[0], size=MAX_SAMPLES_PER_ITER, replace=False)
                Xv = Xv[sel]; Tv = Tv[sel]
            test_acc += net.accuracy(Xv, Tv, batch_size=min(128, Xv.shape[0]))
            total += 1
            del Xv, Tv
        test_acc = (test_acc / max(1, total)) if total > 0 else 0.0

        # ____ GPU 안전 평균 계산 (cupy/numpy 호환) ____
        epoch_train_acc = float(np.mean(np.asarray(train_accs))) if len(train_accs) > 0 else 0.0
        epoch_loss      = float(np.mean(np.asarray(losses)))    if len(losses)  > 0 else 0.0

        train_acc_log.append(epoch_train_acc)
        test_acc_log.append(test_acc)

        print(f"  loss:{epoch_loss:.4f} | train_acc:{epoch_train_acc:.3f} | test_acc:{test_acc:.3f}",
              flush=True)

    # ____ Save ____ 
    # 파라미터 및 로그 저장 (ch4/ch5 train_neuralnet.py 참고)
    out = Path("wordnet_params.npz")
    np.savez(out, **net.params)
    print("Saved:", out.as_posix(), flush=True)

    np.savetxt("train_loss.txt", np.asarray(train_loss_log))
    np.savetxt("train_acc.txt",  np.asarray(train_acc_log))
    np.savetxt("test_acc.txt",   np.asarray(test_acc_log))
    print("Saved logs: train_loss.txt, train_acc.txt, test_acc.txt", flush=True)


if __name__ == "__main__":
    main()
