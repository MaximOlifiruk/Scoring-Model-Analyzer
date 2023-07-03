# Scoring-Model-Analyzer
Scoring-Model-Analyzer is a powerful tool designed for credit scoring data analysis, empowering users with a wide range of operations and functionalities. With this tool, users can easily access, visualize, and normalize data related to credit scoring.

Key Features:

Data Viewing: The tool allows users to view credit scoring data in an intuitive and user-friendly interface. 

Data Visualization: Scoring-Model-Analyzer offers various visualization options to help users gain deeper insights into the credit data. Interactive charts and graphs enable users to identify patterns, trends, and potential relationships within the dataset.

Data Normalization: Ensuring data consistency is crucial for accurate credit scoring. The tool provides data normalization capabilities to standardize the data and bring it to a common scale, enhancing the performance of machine learning models.

Model Selection: Users can leverage pre-built machine learning models tailored for credit scoring. The tool offers a selection of popular algorithms, enabling users to compare model performances and choose the most suitable one for their specific needs.

Model Evaluation: After selecting a machine learning model, users can evaluate its performance on the prepared credit data. 

Ensemble Learning: Users have the option to create ensembles over multiple models.

## Technologies Used:
+ Python
+ Streamlit
+ Numpy
+ Pandas
+ Scikit-learn

## Requirements
+ python
+ pip
+ virtualenv

## Setup to run
1. Download zip file to your local machine
2. Extract the zip file
3. Open terminal/cmd prompt
4. Goto that Path \
   Example: 
```
cd ~/Desktop/Scoring-Model-Analyzer-development/Scoring-Model-Analyzer-development
```

5. Create a new virtual environment in that directory

```
python -m pip install virtualenv
python -m venv myenv
```

6. Activate virtual environment
```
.\myenv\Scripts\activate
```

7. Command line to install all dependencies
```
pip install -r requirements.txt
```

8. Then
```
cd src
```

9. Command line to run your program
```
streamlit run main.py
```

10. Now open your browser and go to this address
```
http://localhost:8501
```