from setuptools import setup

modules = ['networks', 'datahandler', 'trainer']
setup(name='tripletnet-classifier',
      version='2.0',
      description='Deep Triplet Network',
      url='https://github.com/Lucashsmello/TripletNet-on-ESP',
      author='Lucas Mello',
      author_email='lucashsmello@gmail.com',
      license='public',
      py_modules=["tripletnet.%s" % m for m in modules],
      install_requires=[
          "siamese-triplet",
          "torch",
          "torchvision",
          "scikit-learn >= 0.22",
          "pandas >= 0.24",
          "skorch >= 0.8"
      ],
      zip_safe=False)
