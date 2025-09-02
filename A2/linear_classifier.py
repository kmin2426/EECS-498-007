"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


# Template class modules that we will use later: Do not edit/modify this class
from abc import ABC, abstractmethod

class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    input
    - W(D,C): weight vectors (D: Dimension, C: Class)
    - X(N,D): minibatch data (N: sample, D: Dimension)
    - y(N,): index of answer class ex) y = [3, 0, 7, 2, ...]
    - reg(float): L2 Regularization strength

    returns
    - Loss of torch scalar
    - Gradient of loss with respect to W for W -= (learning rate)*dW
    """

    dW = torch.zeros_like(W) # initialize the gradient as zero
    num_classes = W.shape[1] # num of class : C
    num_train = X.shape[0] # size of batch : N
    loss = 0.0 # accumulate loss (float)

    # i = sample index
    for i in range(num_train):
        scores = W.t().mv(X[i]) # z = w^T * X_i (all of scores)
        correct_class_score = scores[y[i]] # correct class's score : s_{y_i}

        # j = class index
        for j in range(num_classes): 
            # except answer class
            if j == y[i]: 
                continue

            margin = scores[j] - correct_class_score + 1  #delta = 1
            
            # max(0, s_j - s_y_i + 1)
            if margin > 0: 
                loss += margin
                dW[:,j] += X[i] # not correct column
                dW[:,y[i]] -= X[i] # correct column

    # mean
    loss /= num_train
    dW /= num_train

    # L2 Regularization
    loss += reg * torch.sum(W * W) 
    dW += 2*reg*W 
    return loss, dW


def svm_loss_vectorized(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float):
    """
    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)

    scores = X.mm(W) # (N,C)

    N = X.shape[0]
    y = y.long() # indexing type long

    correct_scores = scores[torch.arange(N, device=X.device), y].unsqueeze(1) # (N, 1)

    margins = torch.clamp(scores - correct_scores + 1.0, min=0.0) # max(0,s_j - s_y_i + 1)
    margins[torch.arange(N, device=X.device), y] = 0.0 # j=y[i] is always 0

    loss = margins.sum() / N
    loss += reg * torch.sum(W * W)

    coeff = (margins > 0).to(W.dtype) # if margin>0 else 0
    row_sum = coeff.sum(dim=1)
    coeff[torch.arange(N, device=X.device), y] = -row_sum

    dW = X.t().mm(coeff) / N
    dW += 2.0 * reg * W

    return loss, dW



def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    # 0 이상 num_train 미만 1차원 벡터 (1, batch_size)
    idx = torch.randint(0, num_train, (batch_size,), device = X.device)

    # idx : index -> idx 안에 든 번호에 해당하는 행을 X에서 찾아냄
    X_batch = X[idx]
    y_batch = y[idx]

    return X_batch, y_batch



#### Optimization
from typing import Callable

### Optimization
def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape

    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = [] # 학습 과정 기록

    for it in range(num_iters):
        # Implement Sample Batch
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        # Update the weights using the gradient and the learning rate. 
        W -= learning_rate * grad

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


### 학습이 끝난 모델을 이용해서 주어진 입력 X가 어떤 클래스에 속하는지 추정하는 과정
def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    '''
    w (D,C) : 학습이 끝난 weight matrix
     - D: feature 차원 ex) 32x32x3 = 3072
     - C: class 개수 ex) CIFAR-10
    X (N,D) : 입력 데이터
     - N: 샘플 개수
     - D: feature 차원 (W의 D와 일치해야함)
     
    y_pred (N,) : 각 샘플에 대해 가장 점수가 높은 클래스의 인덱스 
    
    '''
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)   
    
    # Implement this method. Store the predicted labels in y_pred. 
    scores = X.mm(W) # XW (N,C) 
    y_pred = scores.argmax(dim=1) # y_pred (N,)

    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    
    # 학습률, regularization 상수 설정
    learning_rates = [5e-3, 6e-3, 7e-3]
    regularization_strengths = [5e-2, 5.5e-2, 6e-2, 6.5e-2, 7e-2]

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.

    # 모델 학습
    cls.train(data_dict["X_train"], data_dict["y_train"], 
              learning_rate = lr, reg = reg,
              num_iters = num_iters, batch_size = 200, verbose = False)
    
    # 학습 데이터로 예측하기
    y_train_pred = cls.predict(data_dict["X_train"])

    # 검증 데이터로 예측하기
    y_val_pred = cls.predict(data_dict["X_val"])

    # 정확도 계산
    train_acc = (y_train_pred == data_dict["y_train"]).float().mean().item()
    val_acc = (y_val_pred == data_dict["y_val"]).float().mean().item()

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Inputs:
    - W (D,C): 각 클래스의 가중치
    - X (N,D): minibatch of data.
    - y (N,): 각 샘플의 정답 클래스 인덱스
    - reg: (float) L2 Reg

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)
    
    # http://cs231n.github.io/linear-classify/)
    N, D = X.shape
    _, C = W.shape
    
    for i in range(N):
        scores = X[i] @ W
        scores = scores - scores.max()
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum()
        
        loss += -torch.log(probs[y[i].long()])
        
        for j in range(C):
            coeff = probs[j]
            if j == int(y[i]):
                coeff -= 1.0
            dW[:,j] += coeff * X[i]
    loss /= N
    dW /= N
    
    loss += reg * torch.sum(W*W)
    dW += 2.0 * reg * W
    
    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    N = X.shape[0]
    y_idx = y.to(dtype=torch.long, device=X.device)

    # 1) scores + numeric stability
    scores = X.mm(W)                                  # (N, C)
    scores = scores - scores.max(dim=1, keepdim=True).values

    # 2) softmax probs
    exp_scores = torch.exp(scores)
    probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (N, C)

    # 3) loss = mean(-log p_y) + reg * ||W||^2
    row_idx = torch.arange(N, device=X.device)
    correct_logprobs = -torch.log(probs[row_idx, y_idx])      # (N,)
    loss = correct_logprobs.mean()
    loss += reg * torch.sum(W * W)                            # 1/2 없음

    # 4) gradient = X^T @ (probs - one_hot(y)) / N + 2*reg*W
    dscores = probs.clone()
    dscores[row_idx, y_idx] -= 1.0                            # (N, C)
    dW = X.t().mm(dscores) / N                                # (D, C)
    dW += 2.0 * reg * W

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = [1e-2, 1e-1, 1e0]
    regularization_strengths = [1e-5, 1e-4, 1e-3]

    return learning_rates, regularization_strengths
