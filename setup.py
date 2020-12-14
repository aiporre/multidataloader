from setuptools import setup, find_packages
from pkg_resources import parse_requirements
requirements = [
    "tensorflow==2.3.1",
    "Pillow==8.0.1",
    "scikit-video==1.1.11"
]
setup(name='dapipe',
      version='0.2.1',
      description='Creates dataset builder objects',
      url='https://github.com/aiporre/multidataloader.git',
      download_url='https://github.com/aiporre/multidataloader/archive/v0.2.1.tar.gz',
      author='Ariel Iporre',
      author_email='ariel.iporre.rivas@gmail.com',
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      install_requires=requirements,
      license='Apache License Version 2.0 ',
      packages=find_packages(exclude=["*_dataset"]),
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.7',
      ],
      )