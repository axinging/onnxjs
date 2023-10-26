import onnxruntime as rt
import sys

print("python3 optoffline.py mobilenetv2-12 all")

sess_options = rt.SessionOptions()


def getOptLevelString(optLevel):
    if sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_BASIC:
        return 'basic'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED: 
        return 'extended'
    elif sess_options.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_ALL: 
        return 'all'
    else:
        return 'none'


def getOptLevelFromString(levelStr):
    if levelStr == 'basic':
        return rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif levelStr == 'extended': 
        return rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif levelStr == 'all':
        return rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        return rt.GraphOptimizationLevel.ORT_DISABLE_ALL


# model_name = "mobilenetv2-12"
model_name = sys.argv[1]
levelStr = sys.argv[2]
# Set graph optimization level
sess_options.graph_optimization_level = getOptLevelFromString(levelStr)
# To enable model serialization after graph optimization set this
levelString = getOptLevelString(sess_options.graph_optimization_level)
sess_options.optimized_model_filepath = "./"+model_name+"-"+levelString+".onnx"

session = rt.InferenceSession("./"+model_name+".onnx", sess_options)
