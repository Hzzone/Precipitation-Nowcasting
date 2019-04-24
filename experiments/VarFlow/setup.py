import os
import setuptools


setuptools.setup(
    name='varflow',
    version="1.0",
    author="Xingjian Shi",
    author_email="xshiab@cse.ust.hk",
    packages=setuptools.find_packages(),
    description='Python Wrapper of VarFlow',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/sxjscience/HKO-7/VarFlow',
    install_requires=['numpy'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
