# carecompass
Without having to know the full details of the model, patients can predict the direction of the disease through the model, build medical data, train verifiable models, protect private data for users, and make verifiable judgments about symptoms to provide an effective care compass

By training UCI datasets using XGBoost classification, we hope to find patterns of inflammation and hope to construct verifiable medical care projects to help patients with symptoms diagnose in time.

Through this work, we also hope to improve the diagnostic accuracy and speed of doctors for these two types of inflammation. We believe that through scientific data analysis methods, we can better understand the patterns of inflammation and provide more accurate predictions and more effective treatment plans.

## Operating guide
```shell
git clone https://github.com/HappyTomatoo/carecompass.git
```
## Environment
```shell
pip install -r requirements.txt
```
## Generate ONNX model
```shell
cd carecompass
# More information in Notebook(Acute_Inflammations_XGBoost_Classification_model.ipynb).
# Export onnx
python Acute_Inflammations_XGBoost_Classification_model.py

```

## Execute transpile
```shell
giza transpile acute_inflammation_xgboost.onnx
```

## Deployments model
```shell
giza deployments deploy --model-id <YOUR_NEW_MODEL_ID> --version-id <YOUR_NEW_VERSION_ID>
```

## Execute predict
```shell
# More information in Notebook(inflammations_predict_onnx.ipynb).
python inflammations_predict_onnx.py

```
An AI Action SDK error message was received.
The Cairo contract can be executed in `carecompass/acute_inflammation_old` to complete execution, like: `scarb cairo-run --available-gas 9999999999`

## Download proofs
```shell
giza deployments download-proof --model-id <MODEL_ID> --version-id <VERSION_ID> --deployment-id <DEPLOYMENT_ID> --proof-id <PROOF_ID> --output-path <OUTPUT_PATH>
```

## Execute verification in local
```shel
giza verify --proof PATH_OF_THE_PROOF
```