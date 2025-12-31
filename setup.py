from setuptools import setup, find_packages

setup(
    name="scimlstudio",
    version="0.0.1",
    packages=find_packages(include=["scimlstudio*"]),
    python_requires=">=3.11",
    install_requires=[
        "torch",
    ]
)
