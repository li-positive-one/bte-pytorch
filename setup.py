from setuptools import setup

setup(
        name='bte-pytorch',     # 包名字
        version='0.0.0',   # 包版本
        description='This is a test of the package',   # 简单描述
        author='Zhengyi Li',  # 作者
        author_email='lizhengyi.pku@gmail.com',  # 作者邮箱
        url='https://github.com/li-positive-one/bte-pytorch',      # 包的主页
        packages=['bte'],                 # 包
        python_requires='>=3.7',
)