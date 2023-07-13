"""
@Description: Using regularized methods for regression
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-24 16:13:47
"""

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

ridge = Ridge(alpha=1)

lasso = Lasso(alpha=1.0)

ela_net = ElasticNet(alpha=1.0, l1_ratio=.5)
