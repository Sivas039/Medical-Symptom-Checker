
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sympcheck-plus",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered healthcare assistant for symptom assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sympcheck-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "sympcheck=gradio_app:main",
            "sympcheck-build-db=build_database:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sympcheck": [
            "data/*.json",
            "prompts/*.txt",
            "static/*",
        ],
    },
    zip_safe=False,
)
