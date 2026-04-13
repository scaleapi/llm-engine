from setuptools import find_packages, setup

setup(
    name="scale-llm-engine",
    python_requires=">=3.9",
    version="0.0.0.beta45",
    packages=find_packages(),
    package_data={"llmengine": ["py.typed"]},
)
