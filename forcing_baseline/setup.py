from setuptools import find_packages, setup

setup(
    name="dreamidv_forcing",
    version="0.1.0",
    description="Causal-Forcing post-training for DreamID-V streaming face swapping",
    packages=find_packages(exclude=("checkpoints", "dataset", "outputs", "scripts")),
    python_requires=">=3.10",
)
