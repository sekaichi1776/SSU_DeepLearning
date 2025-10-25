# coding: utf-8
# ============================================================
#  WordNet 모델 정의 (CNN 기반)
# ============================================================
from __future__ import annotations
from np import np
from collections import OrderedDict

# [1] 활성화 함수 및 손실 함수 [교재 참고 4장 ~ 5장]
def softmax(a: np.ndarray) -> np.ndarray:
    """softmax 계산 (오버플로 방지 포함)"""
    if a.ndim == 2:
        a = a - np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a, axis=1, keepdims=True)
    a = a - np.max(a)
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """교차 엔트로피 오차 계산"""
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    if t.size == y.size:  # one-hot → 인덱스로 변환
        t = np.argmax(t, axis=1)
    batch_size = y.shape[0]
    return float(-np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size)

# [2] 기본 레이어 (ReLU, Affine, SoftmaxWithLoss) [교재 참고: 5장 오차역전파법]
class Relu:
    """ReLU 활성화 함수"""
    def __init__(self): self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy(); out[self.mask] = 0
        return out
    def backward(self, dout):
        dout = dout.copy(); dout[self.mask] = 0
        return dout

class Affine:
    """Affine (완전연결층)"""
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W=W; self.b=b
        self.x=None; self.dW=None; self.db=None; self.original_x_shape=None
    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return self.x @ self.W + self.b
    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.original_x_shape)

class SoftmaxWithLoss:
    """softmax + cross-entropy 결합층"""
    def __init__(self): self.y=None; self.t=None; self.loss=None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, t)
        return self.loss
    def backward(self, dout=1.0):
        if self.t.ndim == 1:
            batch_size = self.t.shape[0]
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size
            return dx * dout
        return (self.y - self.t) / self.t.shape[0] * dout

# [3] im2col / col2im
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """입력 이미지를 2차원 배열로 전개하여 합성곱을 행렬 연산으로 구현"""
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    return col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """im2col로 전개된 배열을 원래 이미지 형태로 복원"""
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]

# [4] Convolution / Pooling Layer [교재 참고: 7장]
class Convolution:
    """합성곱 계층 (im2col 기반)"""
    def __init__(self, W: np.ndarray, b: np.ndarray, stride=1, pad=0):
        self.W=W; self.b=b; self.stride=stride; self.pad=pad
        self.x=None; self.col=None; self.col_W=None; self.dW=None; self.db=None
    def forward(self, x):
        FN, C, FH, FW = self.W.shape; N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = col @ col_W + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        self.x=x; self.col=col; self.col_W=col_W
        return out
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = (self.col.T @ dout).transpose(1,0).reshape(FN, C, FH, FW)
        dcol = dout @ self.col_W.T
        return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

class Pooling:
    """풀링 계층 (max pooling)"""
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h=pool_h; self.pool_w=pool_w; self.stride=stride; self.pad=pad
        self.x=None; self.arg_max=None; self.col=None
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        self.x=x; self.arg_max=arg_max; self.col=col
        return out
    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h*self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(self.col.shape)
        return col2im(dmax, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

# [5] WordNet (CNN 기반 단어 분류기) [교재 참고 7장 SimpleConvNet 구조 기반]
class WordNet:
    """CNN 기반 단어 분류 네트워크"""
    def __init__(self, vocab_size: int, weight_init_std: float = 0.01):
        input_dim=(1,28,28)
        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
        hidden_size=100; output_size=vocab_size
        filter_num = conv_param['filter_num']; filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']; filter_stride=conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = int((input_size - filter_size + 2*filter_pad)/filter_stride + 1)
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 파라미터 초기화
        self._params = {}
        self._params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self._params['b1'] = np.zeros(filter_num)
        self._params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self._params['b2'] = np.zeros(hidden_size)
        self._params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self._params['b3'] = np.zeros(output_size)

        # 계층 구성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self._params['W1'], self._params['b1'],
                                           stride=filter_stride, pad=filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self._params['W2'], self._params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self._params['W3'], self._params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = np.argmax(self.predict(tx), axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        """역전파를 통해 각 파라미터의 기울기 계산"""
        self.loss(x, t)
        dout = self.last_layer.backward(1.0)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    @property
    def params(self):
        """현재 네트워크의 모든 파라미터 접근"""
        return {"W1": self.layers['Conv1'].W, "b1": self.layers['Conv1'].b,
                "W2": self.layers['Affine1'].W, "b2": self.layers['Affine1'].b,
                "W3": self.layers['Affine2'].W, "b3": self.layers['Affine2'].b}
