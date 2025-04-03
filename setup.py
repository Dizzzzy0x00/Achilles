from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='achilles',
    version='1.2',
    description='Static vulnerability detection using machine learning & magic!',
    license="MIT",
    long_description=long_description,
    author='Nicholas Saccente',
    author_email='nicholas.j.saccente@gmail.com',
    url="https://github.com/strickolas/achilles",
    install_requires=['pandas', 'numpy', 'keras', 'tensorflow', 'numpy', 'sklearn', 'javalang']
)
