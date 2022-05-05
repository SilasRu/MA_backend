from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()
    f.close()


setup(
    name='transcript_analyser',
    version='0.0.1',
    author='Zhivar Sourati',
    description='This library provides certain tools to analyze the transcript of the meetings.',
    author_email='zhivarsourati@gmail.com',
    package_dir={'': '.'},
    packages=['transcript_analyser'],
    install_requires=requirements
)
