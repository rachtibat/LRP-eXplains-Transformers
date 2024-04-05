import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='lxt',  
    version='0.5',
    install_requires=[
        'torch',
        'transformers',
        'accelerate',
        'tabulate',
        'matplotlib',
        'bitsandbytes'
    ],
    author="Reduan Achtibat",
    license='BSD 3-clause',
    description="LRP explains Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rachtibat/LRP-for-Transformers",
    packages=setuptools.find_packages(),
)