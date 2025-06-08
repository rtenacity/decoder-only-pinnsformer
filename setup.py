from setuptools import setup, find_packages

setup(
    name="model_components",
    version="0.1.0",
    description="Re-usable PDE model components",
    author="Your Name",
    packages=find_packages(),             # finds the model_components package
    include_package_data=True,            # include data files if you add them
    install_requires=[                    # any dependencies go here
        # e.g. "numpy", "torch>=1.8"
    ],
    zip_safe=False,
)
