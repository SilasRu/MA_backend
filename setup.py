from setuptools import setup


setup(
    name='Transcript_Analysis',
    version='0.0.1',
    author='Zhivar Sourati',
    description='This library provides certain tools to analyze the transcript of the meetings.',
    author_email='zhivarsourati@gmail.com',
    package_dir={'': 'Transcript_Analysis'},
    packages=['abstractive.*', 'extractive.*', 'utils.*', 'data_types.*']
)
