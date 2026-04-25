from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="cism",
    version="0.1.0",
    description="Context-dependent Identification of Spatial Motifs",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(include=["cism", "cism.*", "pairwise", "pairwise.*"]),
    py_modules=["motif_hits_from_csv"],
    include_package_data=True,
    package_data={
        "cism": [
            "FANMOD_binaries/*",
            "FANMOD_binaries/**/*",
        ],
    },
    install_requires=[
        "alphashape",
        "dotmotif",
        "joblib",
        "matplotlib",
        "networkx",
        "numpy",
        "opencv-python",
        "optuna",
        "pandas",
        "scikit-learn",
        "scipy",
        "shap",
        "shapely",
        "torch",
        "torch-geometric",
        "tqdm",
    ],
    zip_safe=False,
)
