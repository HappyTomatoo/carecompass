from giza_actions.model import GizaModel
from sklearn import preprocessing
from giza_actions.task import task
from giza_actions.action import Action, action
import numpy as np


MODEL_ID = 340  # Model ID 325 340
VERSION_ID = 5  # Version ID

prediction_result_dict = {0:'Normal', 
                          1:'Inflammation of urinary bladder', 
                          2:'Nephritis of renal pelvis origin', 
                          3:'Inflammation of urinary bladder and Nephritis of renal pelvis origin'
                         }

def parser_acute_inflammations(line: str) -> str:
    line_list = line.replace('no', '0').replace('yes', '1').replace(
        '\n', '').replace(',', '.').split('\t')
    
    return '\t'.join(line_list)

@task(name=f'Preprocess input')
def preprocess_input(origin_input: str):
    preprocess_input = []
    # preprocess data
    origin_input = parser_acute_inflammations(origin_input)
    
    # normalization 0-1
    with open("column_normalization_info.tsv", 'r') as f:
        column_max_and_min = [line.replace('\n', '').split('\t')[1:] for line in f.readlines()]
        for index in range(len(column_max_and_min[0])):
            train_max = float(column_max_and_min[0][index])
            train_min = float(column_max_and_min[1][index])
            prediction_data = float(origin_input.split('\t')[index])
            normalized_prediction = (prediction_data - train_min) / (train_max - train_min)
            preprocess_input.append(normalized_prediction) 
    
    return preprocess_input 


@task(name=f'Prediction with ONNX')
def prediction(input, model_id, version_id):
    XGBoost_model = GizaModel(id=model_id, version=version_id)
    # XGBoost_model = GizaModel(model_path = "./acute_inflammation.onnx")
    input = np.array([input])
    (result, result_probs) = XGBoost_model.predict(input_feed = {"input": input}, verifiable=False, output_dtype="Span<u32>, Tensor<FP16x16>")
    print(result, input)
    return result, result_probs

@action(name=f'Execution: Prediction with ONNX', log_prints=True )
def execution():
    origin_input = "36.6	no	no	yes	yes	yes"
    prep_input = preprocess_input(origin_input)
    print("prep_input",prep_input)
    (predicted_digit,predicted_probs) = prediction(prep_input, MODEL_ID, VERSION_ID)
    print("Prediction output:", predicted_digit, predicted_probs)
    # print(f"Predicted result: {prediction_result_dict[predicted_digit]}")
    return predicted_digit

execution()