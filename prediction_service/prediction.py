import yaml
import os
import json
import joblib
import numpy as np


params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)



def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)

    
    try:
        prediction = model.predict(data).tolist()[0]
        #prediction = model.predict([[7.7,0.56,0.08,2.5,0.114,14.0,46.0,0.9971,3.24,0.66,9.6]])
        if 3 <= prediction <= 8:
            return prediction
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema

def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()

        if not (schema[col]["min"] <= float(dict_request[col][0]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    
    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
#        print("data before list comp")
        data = [list(map(float, i)) for i in list(data)]
#        print("data after list comp\n\n",data,"\n\n",type(data),np.array(data).reshape(1,-1))       
        response = predict(np.array(data).reshape(1,-1))
        return response
    else:
        print("validation failed")

def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = np.array([list(map(float, i)) for i in list(data)]).reshape(1,-1)
            print(data)
            response = predict(data)
            response = {"response": response}
            return response
            
    except NotInRange as e:
        response = {"the_exected_range": get_schema(), "response": str(e) }
        return response

    except NotInCols as e:
        response = {"the_exected_cols": get_schema().keys(), "response": str(e) }
        return response


    except Exception as e:
        response = {"response": str(e) }
        return response

