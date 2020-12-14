from setuptools import setup, find_packages
from pkg_resources import parse_requirements
parsed_requirements = parse_requirements('requirements.txt')
requirements = [str(r) for r in parsed_requirements]
setup(name='dapipe',
      version='0.3',
      description='Creates dataset builder objects',
      url='https://github.com/aiporre/multidataloader.git',
      download_url='https://github.com/aiporre/multidataloader/archive/v0.3.tar.gz',
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