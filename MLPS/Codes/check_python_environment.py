"""
@Description: check python environment
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-19 22:49:26
"""

import sys
# from distutils.version import LooseVersion
import platform
from packaging import version

python_ver = platform.python_version()
# deprecated warning
# if LooseVersion(sys.version) < LooseVersion('3.8'):
if version.parse(python_ver) < version.parse('3.8'):
    print(
        f'[FAIL] We recommend Python 3.8 or later but found version {sys.version}')
else:
    print(f'[OK] Your Python Version is {sys.version}')


def get_packages(pkgs) -> list:
    versions = []
    for p in pkgs:
        try:
            imported = __import__(p)
            try:
                versions.append(imported.__version__)
            except AttributeError:
                try:
                    versions.append(imported.version_info)
                except AttributeError:
                    versions.append('0.0')
        except ImportError:
            print(f'[FAIL]: {p} is not installed and/or cannot be imported')
            versions.append('N/A')
    return versions


def check_packages(d: dict) -> None:
    versions = get_packages(d.keys())
    for (pkg_name, suggest_ver), actual_ver in zip(d.items(), versions):
        if actual_ver == 'N/A':
            continue
        actual_ver, suggest_ver = version.parse(
            actual_ver), version.parse(suggest_ver)
        if actual_ver < suggest_ver:
            print(
                f'[FAIL] {pkg_name} {actual_ver}, please upgrade to >= {suggest_ver}')
        else:
            print(f'[OK] {pkg_name} {actual_ver}')


if __name__ == '__main__':
    d = {
        'numpy': '1.21.2',
        'scipy': '1.7.0',
        'matplotlib': '3.4.3',
        'sklearn': '1.0',
        'pandas': '1.3.2'
    }
    check_packages(d)
