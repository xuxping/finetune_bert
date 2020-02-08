"""
install:
    pip3 install .
"""

from setuptools import setup, find_packages

from finetune import __version__

with open('README.md', 'r', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="finetune",
    version=__version__,
    author="mkavim",
    author_email="xuxp2018@163.com",
    description="finetune bert with keras",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords='nlp,finetune,bert',
    python_requires=">=3.5.0",
    license="Apache",
    url="https://github.com/mkavim/finetune_bert",
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]

)
