"""
@Title: Working with data sources
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 17:21:45
@Description: 
"""

import tensorflow_datasets as tfds
# When you are importing a dataset for the first time, a bar will
# point out where you are as you download the dataset. If you
# prefer, you can deactivate it if you type the following:
# tfds.disable_progress_bar()
iris = tfds.load("iris", split="train")
