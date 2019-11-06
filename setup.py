from setuptools import setup, find_packages

with open('requirements.txt', 'r') as file:
    requirements = list(file.readlines())


setup(
    name='monte_carlo_dropout',
    version='0.1.0',
    url='https://github.com/aredier/monte_carlo_dropout.git',
    author='Author Name',
    author_email='author@gmail.com',
    description='using dropout to infer confidence over a NN output',
    packages=find_packages(),
    install_requires=requirements,
)