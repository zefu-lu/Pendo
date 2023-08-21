from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Search your Notion content in Perplexity style.'
LONG_DESCRIPTION = 'Search your Notion content in Perplexity style.'

setup(
    name="pendo",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Shawn Lu",
    author_email="shawnlu25@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='llm, notion, search',
    classifiers= [
        'License :: OSI Approved :: MIT License',
    ]
)
