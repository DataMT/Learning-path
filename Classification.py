'''
# 感知向量机
# ***.model_selection ,train_test_split, .linear_model ,Perceptron ***

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# 定义特征变量和目标变量 (取values 和 pd 转np的方式都可以获得 np类型的值。那么，哪一种是最好的？
# k：DataFrame中如何一次读取多个列

feature = data[['x', 'y']].values
target = data['class'].values

# 对数据集进行切分，70% 为训练集，30% 为测试集。
# train_test_split, test_size , random_state

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=50)

#检查切分之后的数据集情况
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 构建模型

model = Perceptron(max_iter=1000, tol=1e-3)

# 训练模型

model.fit(X_train, y_train)

# 预测
results = model.predict(X_test)

results

'''
