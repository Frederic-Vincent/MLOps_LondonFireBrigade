from setuptools import setup, find_packages

# Lire requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="MLOps_LondonFireBrigade",
    version="0.1",
    author="Frédéric Vincent",
    author_email="frederic.vincent@protonmail.com",
    description="Un projet MLOps pour la London Fire Brigade",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",  # version minimale de Python requise
)

