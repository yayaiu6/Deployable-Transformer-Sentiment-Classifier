from setuptools import setup, find_packages

setup(
    name="sentiment140_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    author="yahya mahroof",
    author_email="yahyamahroof35@gmail.com",
    description="Utilities for the Sentiment140 Transformer model",
    license="MIT",
)