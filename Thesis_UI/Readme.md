# Machine Learning

# Please maintain this format for proper build process

# This python is suited for MACOS and Ubuntu

# Installation

PYTHON=3.6

The packages required are (eg: requirements.txt)

- pandas
- numpy
- matplotlib==3.1.0
- PyQt6
- seglearn
- scipy==0.19.1
- scikit-learn==0.19.1
- pyinstaller

# Controller.py is the main file for the UI screen

```
python3.6 -m venv ./
source ./bin/activate
pip install -r requirements.txt
python ./src/front_end/controller.py
```

# If you want to edit the files in pyqt editor we need to reflect in respective py file so

# Go to the src/front_end/ui_files

```
pyuic5 MachineLearning.ui > MachineLearning.py
pyuic5 MachineLearningPrediction.ui > MachineLearningPrediction.py
pyuic5 MainWindow.ui > MainWindow.py
pyuic5 PredictionDistance.ui > PredictionDistance.py
```

# Please keep in mind about the **config.json\*** file

`python ./src/front_end/controller.py`

# config.json is the current directory
