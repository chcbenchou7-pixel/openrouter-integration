from setuptools import setup, find_packages

setup(
    name="openrouter-integration",
    version="1.0.0",
    packages=find_packages(include=["openrouter_integration", "openrouter_integration.*"]),
    install_requires=[
        "httpx>=0.27.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.3.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
    },
    author="OpenRouter Integration Team",
    author_email="dev@example.com",
    description="生產級 OpenRouter Python 客戶端 - 統一介面、智能路由、成本控制、Function Calling、結構化輸出",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openrouter-integration",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)