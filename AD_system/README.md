# Auto Diagnosis System

## Usage
All the necessary files are ready in the `stored_files` folder. One thing left to do is to unzip `inquiry_system.zip` to the same folder. 
Therefore, you can skip the following steps to the inference using `AD_system.ipynb`.

#### For data preprocessing,
```
python preprocessing.py <input_path> <--output_path>
```
`input_path`: intput path to the patient data file (csv file)

`output_path`: output path (directory)

#### For passing data through the inquiry system or change the parameters in the inquiry system, please refer to `inquiry_system.py`
```
python inquiry_system.py <input_path> <--output_path>
```
`input_path`: intput path to the patient data file (zst file)

`output_path`: output path (directory)

#### For training the model based on the result of the inquiry system, please refer to `LSTM.ipynb`.

#### For inferencing the model, please refer to `AD_system.ipynb`.
