"""
@Description: run this codes to check results returned
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-03 23:46:05
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
sys.path.append('../')
from models import AdalineSGD
from utils.visualize import plot_decision_regions
from utils.dataset import get_iris

X, y = get_iris()
ada_sgd = AdalineSGD(n_iter=15, eta=.01, random_state=1)
ada_sgd.fit(X, y)
plot_decision_regions(X, y, ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()
