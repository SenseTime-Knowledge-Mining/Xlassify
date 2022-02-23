#
# automatic generated for module implementation
# generate time: 2022-02-21 16:00:00
#
from setuptools import setup, find_packages
# from Cython.Build import cythonize

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Xlassify',
    version='1.0.0',
    keywords=('Xlassify', 'genome', 'microbiome', 'taxonomic', 'bacteria'),
    author_email='maokangkun@pjlab.org.cn',
    author='kangkun',
    description='An alignment-free deep-learning model trained to classify human gut bacteria',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SenseTime-Knowledge-Mining/Xlassify',
    project_urls={
        'Bug Tracker': 'https://github.com/SenseTime-Knowledge-Mining/Xlassify/issues'
    },
    platforms='any',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7.1",
    include_package_data=True,

    package_data={
        '': ['*.ckpt', '*.csv', 'dataset/test/*.fasta'],
    },

    entry_points={
        'console_scripts': [
            'xlassify = xlassify.interface_test:main'
        ]
    },

    install_requires=[
        'numpy>=1.18',
        'pandas>=1.1',
        'torch>=1.7',
        'tqdm',
        'requests'
    ],


    # ext_modules=cythonize([
    #         'src/xlassify/tools/kmer_utils.py',
    #         'src/xlassify/network/model.py',
    #         'src/xlassify/interface_test.py',
    #         'src/xlassify/interface.py',
    # ], language_level=3),

)
