# Part II Project


## File structure
```
├── LICENSE
├── README.md
├── data # Data from third party sources.
├── models          # Trained and serialized models.
├── notebooks       # Jupyter notebooks.
├── figures         # Generated graphics and figures to be used in reporting.
└── src              # Source code for use in this project.
    ├── data         # Data engineering scripts.
    │   ├── build_features.py
    │   ├── cleaning.py
    │   ├── ingestion.py
    │   ├── labeling.py
    │   ├── splitting.py
    │   └── validation.py
    ├── models       # ML model engineering (a folder for each model).
    │   └── model1
    │       ├── dataloader.py
    │       ├── hyperparameters_tuning.py
    │       ├── model.py
    │       ├── predict.py
    │       ├── preprocessing.py
    │       └── train.py
    └── visualization # Scripts to create exploratory and results
        │             # oriented visualizations.
        ├── evaluation.py
        └── exploration.py
```


## License
```
MIT License

Copyright (c) 2024 Simon Luo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```