from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="credit-scorecard-builder",
    version="1.0.0",
    author="Pawan Mishra",
    author_email="pawanbit034@gmail.com",
    description="A comprehensive package for building credit risk scorecards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pawanmishra/credit-scorecard-builder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.2.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "optbinning>=0.8.0",
        "lightgbm>=3.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
        "probatus>=1.4.0",
        "shap>=0.39.0",
        "plotly>=4.14.0",
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'scorecard-builder=scorecard.cli:main',
        ],
    },
) 