# C++17 Data Science and Machine Learning Course

Three connected learning tracks teach C++ foundations, data science and from-scratch machine learning. The project uses C++17 and CMake. Two optional PowerShell scripts run every lesson and verify the generated results. No Python runtime or Python verification dependencies are required.

```text
01_get_to_know/    C++ lessons, debugging, testing and numerical computing
02_data_science/  mathematical bridge, tabular toolkit, advanced EDA, final pipeline
03_ml_course/     supervised/unsupervised algorithms and applied ML projects
docs/          curriculum, notation, data catalog, validation and artifact rules
scripts/       run all lessons and verify artifacts
build/         generated aggregate build
```

## Build and run

Install CMake and a C++17 compiler. On this Windows checkout with MinGW:

```powershell
cmake --preset course
cmake --build --preset course
./scripts/run_all_lessons.ps1          # runs every lesson; files a lesson writes land in results/<stem>_results/
ctest --preset course
./scripts/verify_artifacts.ps1 -RequireFullML
```

On other platforms configure with `cmake -S . -B build -DBUILD_TESTING=ON`, build with `cmake --build build`, then run `ctest --test-dir build --output-on-failure`. The optional preset selects MinGW and is Windows-specific. Each track also supports independent configuration.

All results stay inside their owning module's `results/` directory. ML result subfolders match the C++ source filename, such as `04_end_to_end_results` and `predict_results`. The Titanic capstone uses one shared fitted pipeline for feature definitions, training-only imputation/scaling, evaluation and model reload. K-means, hierarchical clustering and silhouette evaluation are explicit unsupervised priorities.

Read [the project structure](docs/project_structure.md), [the learning path](docs/learning_path.md) (order of study and the per-algorithm format), and [the file guide](docs/file_guide.md) (which files are hand-written, which are generated, and when to create each). An implemented lesson or successful compilation alone does not certify a complete scientific evaluation.

## License

Licensed under the [MIT License](LICENSE). Bundled datasets retain their upstream licenses and attribution, documented in their `PROVENANCE.md` files.