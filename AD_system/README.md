# Auto Diagnosis System

## Run AD System as Streamlit app
```
streamlit run streamlit_app.py
```
![image](https://github.com/zihanzhao1999/CapstoneProject/assets/111836220/e23e3a57-4d3e-48a2-b5f4-7f7ed4ec57d5)
![image](https://github.com/zihanzhao1999/CapstoneProject/assets/111836220/d6843800-6fc6-415b-aa7a-1bfce1e2086a)
![image](https://github.com/zihanzhao1999/CapstoneProject/assets/111836220/19e032f2-2cf9-4fda-8639-e3626343f7f8)


## Development Usage
All the necessary files are ready in the `stored_files` folder. One thing left to do is to unzip `inquiry_system.zip` to the same folder, or you can directly use `simple_tree.pkl` as the decision tree. 
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
