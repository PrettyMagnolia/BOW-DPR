from setuptools import setup, find_packages

setup(
    name='bowdpr',
    version='0.0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'torch==2.1.0',
        'transformers==4.45.2',
        'datasets==3.2.0',
        'faiss==1.9.0',
        'tqdm==4.67.1',
        'fire',
        'pandas',
        'regex==2024.11.6',
        'tensorboard',
        'tabulate',
        'sentence_transformers==3.1.1',
        'numpy==2.2.1',
        'setuptools==75.1.0',
    ],
)