from setuptools import find_packages, setup

setup(
    name="scale-llm-engine",
    python_requires=">=3.8",
    version="0.0.0.beta45",
    packages=find_packages(),
    # types-setuptools 82.0.0+ tightened package_data to _DictLike; the literal dict
    # still works at runtime, only the new stub disagrees. Suppress at the call site
    # rather than down-pinning the stub (which would mask real future tightenings).
    package_data={"llmengine": ["py.typed"]},  # type: ignore[arg-type]
)
