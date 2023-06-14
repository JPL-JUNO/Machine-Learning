"""
@Description: Learning with ensembles
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 21:11:09
"""
import sys
sys.path.append('./')
sys.path.append('../')
from utilsML.funcs import ensemble_error
print(ensemble_error(n_classifier=11, error=.25))

import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.00, 1.01, .01)
ens_errors = [ensemble_error(n_classifier=11, error=error)
              for error in error_range]
plt.plot(error_range, ens_errors, label='Ensemble Error', linewidth=2)
plt.plot(error_range, error_range, linestyle='--',
         label='Base error', linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend()
plt.grid(alpha=0.5)
plt.show()
