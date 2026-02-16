from setuptools import setup, find_packages

setup(
    name="fixed_income_engine",
    version="0.1.0",
    description="Fixed income valuation engine",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
    ],
    python_requires=">=3.8",
)
