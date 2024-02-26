import numpy as np
import logging
from giza_actions.action import Action, action
from giza_actions.task import task
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier as XGBC, XGBRegressor as XGBR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import joblib
import onnx
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

def parser_acute_inflammations(line: str) -> str:
    line_list = line.replace('no', '0').replace('yes', '1').replace(
        '\n', '').replace(',', '.').split('\t')
    new_line_list = line_list[0:-2]
    if line_list[-2] == '0':
        if line_list[-1] == '0':
            new_line_list.append('0')
        else:
            new_line_list.append('2')
    else:
        if line_list[-1] == '0':
            new_line_list.append('1')
        else:
            new_line_list.append('3')
    return '\t'.join(new_line_list) + '\n'
    
@task(name=f'Prepare Datasets')
def prepare_datasets():
    print("Prepare dataset...")
    dataset_path = "data/diagnosis.data"
    with open(dataset_path, encoding="utf-16") as f:
        lines = f.readlines();
        remove_index = []
        for index, line in enumerate(lines):
            line = parser_acute_inflammations(line)
            if len(line):
                lines[index] = [float(ele) for ele in line.replace('\n', '').split('\t')]
            else:
                remove_index.append(index)
        # remove none line
        for index in remove_index:
            lines.remove(lines[index - index_offset])
        x = [line[:-1] for line in lines]
        y = [[int(line[-1])] for line in lines]
        x = np.array(x)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        x = min_max_scaler.fit_transform(x)
        combined_array = np.concatenate((x, y), axis=1)
        print("✅ Datasets prepared successfully")
    
        return x, y

@task(name=f'Training model')
def get_model(x, y, model_type: str = "classification"):
    # Divide the training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    if model_type == "classification":
        XGBmodel = XGBC(n_estimators=10).fit(x_train, y_train)
    else:
        XGBmodel = XGBR(n_estimators=10).fit(x_train, y_train)
    # Get Score
    print("Score", XGBmodel.score(x_test, y_test))
    # Get Mean Square Error
    print("Mean Square Error", MSE(y_test, XGBmodel.predict(x_test)))
    # Get feature importance
    print("Feature importance", XGBmodel.feature_importances_)

    return XGBmodel

@task(name=f'Export to ONNX')
def onnx_export(model, filename, input_size = 6, target_opset =15):
    # Export model as a ONNX
    onnx_model_converted = convert_xgboost(XGBmodel, 'tree-based classifier',
                             [('input', FloatTensorType([1, input_size]))],
                             target_opset=target_opset)
    onnx.save_model(onnx_model_converted, filename)

@action(name=f'Model Development', log_prints=True )
def develop_model():
    x, y = prepare_datasets()
    XGBmodel = get_model(x, y)

    # Convert to ONNX
    onnx_export(XGBmodel, "acute_inflammation_xgboost.onnx", 6, 15)

if __name__ == "__main__":
    action_deploy = Action(entrypoint=develop_model, name="acute_inflammation_xgboost_action")
    action_deploy.serve(name="acute_inflammation_xgboost_deployment")
