from setuptools import setup

modules=['networks','classifiers.augmented_classifier', 'datahandler', 'feature_selector', 'trainer']
setup(name='tripletnet-classifier',
    version='0.5',
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
		"pandas >= 0.24"
    ],
    zip_safe=False)
