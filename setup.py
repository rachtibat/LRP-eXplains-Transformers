import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='lxt',  
    version='0.6.1',
    install_requires=[
        'torch<=2.1.0',
        'transformers>=4.46.2',
        'accelerate',
        'tabulate',
        'matplotlib',
        'bitsandbytes',
        'open_clip_torch',
    ],
    author="Reduan Achtibat",
    license='BSD 3-clause',
    description="LRP explains Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rachtibat/LRP-for-Transformers",
    packages=setuptools.find_packages(),
)
