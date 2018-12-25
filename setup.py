from setuptools import setup, find_packages

exec(open('loop/version.py').read())

requirements = [
    'psutil',
    'tqdm',
    'matplotlib',
    'numpy',
    'pandas',
    'torch',
    'torchvision'
]

dev_requirements = {'dev': [
    'pytest'
]}

setup(
    name='torch-loop',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extra_require=dev_requirements,
    tests_require=['pytest']
)
