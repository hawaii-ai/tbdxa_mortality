from setuptools import setup, find_packages

setup(name='TBDXAMortalityPrediction',
      version=0.1,
      author='Yannik Glaser',
      author_email='yglaser@hawaii.edu',
      description='Deep learning for mortality prediction from TBDXA',
      packages=find_packages(),
      license='cc-by-nc-sa 4.0',
      include_package_data=True,
      install_requires=[
          'tensorflow >= 2.2.0', 'tensorflow-addons >= 0.13.0',
          'numpy >= 1.18.5', 'pandas >= 1.2.4', 'imageio >= 2.9.0',
          'scikit-learn >= 0.24.2', 'lifelines >= 0.9.4'
      ])
