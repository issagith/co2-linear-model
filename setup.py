from setuptools import setup, find_packages

setup(
    name="linearmodel",
    version="0.1.0",
    author="KA Issa",
    description="Un package Python pour l'analyse statistique et la régression linéaire",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.7",
)
