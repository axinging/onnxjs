import onnx
import sys
import onnx, onnx.numpy_helper
import json
from pprint import pprint
from graph_utils import tensor_dtype_to_json_dtype
import numpy as np
from google.protobuf.json_format import MessageToJson


def saveObjectToJsonFile(dictionary, name):
    # Serializing json
    json_object = json.dumps(dictionary, indent=2)
    
    # Writing to sample.json
    with open(name, "w") as outfile:
        outfile.write(json_object)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

model = onnx.load('./albert-base-v2-all.onnx')
#weights = model.graph.initializer
#print(weights)

#[tensor] = [t for t in model.graph.initializer if t.name == "onnx::MatMul_1413"]
#print(tensor)
for t in model.graph.initializer:
    print(t.name)
    print(t.dims)
    print(list(t.dims))
    print(onnx.helper.tensor_dtype_to_string(t.data_type)) # onnx.helper.tensor_dtype_to_storage_tensor_dtype
    print(onnx.helper.tensor_dtype_to_storage_tensor_dtype(t.data_type)) # 
    print(tensor_dtype_to_json_dtype(t.data_type))
    w = onnx.numpy_helper.to_array(t).flatten()
    json_outs =  {"data":w.tolist(),"dims": list(t.dims),"type":tensor_dtype_to_json_dtype(t.data_type)}
    #json_data = json.dumps(json_outs, escape_forward_slashes=False)

    print("JSON Data  ")
    # print(json_data)
    saveObjectToJsonFile(json_outs, "./temp/"+t.name.replace(":", "_")+".json")


print(w)

