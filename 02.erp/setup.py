from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="erp-processor",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Tools for processing and analyzing ERP data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omare32/erp-data-processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "openpyxl>=3.0.9",
        "python-dotenv>=0.19.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "pyodbc>=4.0.30",
        "pytz>=2021.3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.12b0",
            "isort>=5.10.1",
            "mypy>=0.931",
            "flake8>=4.0.1",
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "erp-processor=erp_processor.cli:main",
        ],
    },
)
