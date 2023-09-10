"""
@Title: 批量修改文件名
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 10:00:27
@Description: 
"""
import os

for file in os.listdir('./'):
    os.rename(file, file.replace(" ", "_").lower())
