from setuptools import setup, find_packages

setup(
    name="scimlstudio",
    version="1.6.3",
    packages=find_packages(include=["scimlstudio*"]),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.9.0",
        "gpytorch>=1.15.1",
        "botorch>=0.16.1"
    ]
)
