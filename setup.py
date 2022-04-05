from pathlib import Path
from setuptools import find_packages, setup

setup(
    name='archetypal-analysis-popgen',
    version='0.1.0',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    description='Archetypal Analysis for Population Genetics',
    url='https://github.com/AI-sandbox/archetypal-analysis',
    author='Júlia Gimbernat-Mayol',
    author_email='juliagimbernat@gmail.com',
    entry_points={
        'console_scripts': ['archetypal-analysis=entry:main']
    },
    license='CC BY-NC 4.0',
    packages=find_packages('archetypal_analysis/')+['.'],
    package_dir={"": "archetypal_analysis"},
    python_requires=">=3.6",
    install_requires=["matplotlib==3.3.4",
                      "numpy==1.19.5",
                      "pandas==1.1.5",
                      "pandas-plink==2.2.4",
                      "scikit-allel==1.3.5",
                      "scikit-learn==0.24.2",
                      "scipy==1.5.4"
                      ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.9',
    ],
)
