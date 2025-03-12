from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phishing-domain-detection",
    version="1.0.0",
    author="Kawaki-1998",
    author_email="your.email@example.com",
    description="A machine learning-based phishing domain detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kawaki-1998/phising_detect",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "lightgbm>=3.2.1",
        "mlflow>=1.20.0",
        "joblib>=1.0.1",
        "python-multipart>=0.0.5",
        "pydantic>=1.8.2",
    ],
    entry_points={
        "console_scripts": [
            "phishing-detector=src.api.app:main",
        ],
    },
) 