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
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.2",
        "lightgbm>=4.1.0",
        "mlflow>=2.8.1",
        "joblib>=1.3.2",
        "python-multipart>=0.0.6",
        "pydantic>=2.5.2",
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.2",
    ],
    entry_points={
        "console_scripts": [
            "phishing-detector=src.api.app:main",
        ],
    },
) 