from setuptools import setup, find_packages
requirements = [
                'tensorflow==2.1.0',
                'Pillow==7.0.0',
                'scikit-video==1.1.11'
                ]
setup(name='datapipe',
      version='0.1',
      description='Creates dataset builder objects',
      url='https://github.com/aiporre/video_tools.git',
      author='Ariel Iporre',
      author_email='ariel.iporre.rivas@gmail.com',
      long_description=open('README.md').read(),
      install_requires=requirements,
      license='Apache License Version 2.0 ',
      packages=find_packages(exclude=["*_dataset"]),
      zip_safe=False,
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: Apache License 2.0',
          'Programming Language :: Python :: 3.7',
      ],
      )