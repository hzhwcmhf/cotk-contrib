from setuptools import setup, find_packages

setup(
    name='cotk-contrib',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Conversational Toolkits Contribution',
    long_description=open('README.md', encoding='UTF-8').read(),
    install_requires=[
        'numpy>=1.13',
        'nltk>=3.2'
    ],
    url='https://github.com/hzhwcmhf/cotk-contrib',
    author='hzhwcmhf',
    author_email='hzhwcmhf@gmail.com'
)
