from setuptools import setup

modules = ['networks', 'classifiers.augmented_classifier', 'datahandler', 'trainer']
setup(name='tripletnet-classifier',
      version='1.2.3',
      description='Deep Triplet Network',
      url='https://github.com/Lucashsmello/TripletNet-on-ESP',
      author='Lucas Mello',
      author_email='lucashsmello@gmail.com',
      license='public',
      py_modules=["tripletnet.%s" % m for m in modules],
      install_requires=[
          "siamese-triplet",
          "torch == 1.6.0",
          "torchvision == 0.7.0",
          "scikit-learn == 0.22.2",
          "pandas >= 0.24",
          "Pillow == 7.1.2"
      ],
      zip_safe=False)
