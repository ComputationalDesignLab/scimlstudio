from setuptools import setup, find_packages

setup(
    name="scimlstudio",
    version="1.0.0",
    packages=find_packages(include=["scimlstudio*"]),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.9.0",
    ]
)
