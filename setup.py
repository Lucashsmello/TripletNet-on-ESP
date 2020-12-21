from setuptools import setup

modules = ['networks', 'datahandler', 'trainer', 'callbacks','classifiers.TorchBaggingClassifier', 'TripletNetClassifierMCDropout', 'utils']
setup(name='tripletnet-classifier',
      version='2.7',
      description='Deep Triplet Network',
      url='https://github.com/Lucashsmello/TripletNet-on-ESP',
      author='Lucas Mello',
      author_email='lucashsmello@gmail.com',
      license='public',
      py_modules=["tripletnet.%s" % m for m in modules],
      install_requires=[
          "siamese-triplet @ git+https://github.com/Lucashsmello/siamese-triplet",
          "torch",
          "torchvision",
          "scikit-learn >= 0.22",
          "pandas >= 0.24",
          "skorch >= 0.8"
      ],
      zip_safe=False)
