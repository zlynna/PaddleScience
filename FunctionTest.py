import sys
sys.path.insert(0, 'utils')
import paddle
sys.path.insert(0, 'source')
from GCNNModel import e2vcg2connectivity, PossionNet, TestNet
from paddle.nn import initializer
from paddle.nn import Linear

# x = paddle.to_tensor([0, 1, 2, 3])
# index = paddle.to_tensor([[[1], [2], [3], [1]]])
# updates = paddle.to_tensor([9, 10, 11, 12])

# print(paddle.scatter_nd_add(x, index, updates))
x = paddle.zeros(shape=[2, 2,1], dtype='float32')
updates = paddle.ones(shape=[3, 2,1], dtype='float32')
index = paddle.to_tensor([[1,1],
                        [0,0],
                        [1,1]], dtype='int64')

output = paddle.scatter_nd_add(x, index, updates)
print(output)