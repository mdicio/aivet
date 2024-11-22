from setuptools import setup, find_packages

setup(
    name="bloodtest_analysis",  # Name of your project
    version="0.1.0",  # Initial version
    description="A library for analyzing blood test results with LLM integration.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your_email@example.com",
    url="https://github.com/yourusername/bloodtest_analysis",  # Update with your repo link if available
    packages=find_packages(
        include=["modules", "modules.*"]  # Include the core modules in the package
    ),
    package_data={
        # Include non-Python files like models, or config templates
        "": ["models/**/*.safetensors", "models/**/*.md"]
    },
    install_requires=[
        # Add dependencies here
        "numpy",
        "pandas",
        "torch",  # For handling large models
        "fastapi",  # If you plan to use the FastAPI web app
        "uvicorn",  # For running the FastAPI server
        "Pillow",  # For image handling
    ],
    extras_require={
        "dev": ["pytest", "flake8"],  # Development dependencies
    },
    entry_points={
        "console_scripts": [
            # Example CLI tool bindings, if needed
            "bloodtest-run=orchestrator.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Adjust based on your project's requirements
    include_package_data=True,  # Ensures non-code files are included
    zip_safe=False,
)
