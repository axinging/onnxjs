import Long from 'https://cdn.jsdelivr.net/npm/long@5.2.3/index.js';
import * as onnxdecoder from './onnxdecoder.js';

const onnx = ort.OnnxProto.onnx;  // onnxModule.onnx;

function sleep(ms) {
  let start = new Date().getTime();
  let expire = start + ms;
  while (new Date().getTime() < expire) {
  }
  return;
}

export function writeObjectToFile(json_object, name, time = 200) {
  let object = json_object;
  const file_name = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  const file = new Blob([JSON.stringify(object)], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = file_name;
  a.click();
  sleep(time);
}

export function writeMapToFile(json_object, name, time = 200) {
  let object = json_object;
  const file_name = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    // object = Object.fromEntries(object);
    for (let [key, value] of object) {
      console.log(key + ' = ' + value);
      writeObjectToFile(value, key + '.json');
    }
  }
}

export function writeObjectToFile2(json_object, name, time = 200) {
  let object = json_object;
  const file_name = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  const file = new Blob([object], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = file_name;
  a.click();
  sleep(time);
}

export async function readObjectFromJson(fileUrl) {
  const response = await fetch(fileUrl);
  // const blob = await response.blob();
  let blobObject;
  try {
    blobObject = await response.json();  // JSON.parse(await blob.text());
  } catch (err) {
    console.error(err);
  }

  try {
    const text = await response.text();
    console.log(JSON.parse(text));
  } catch (err) {
    console.error(err);
  }
  return blobObject;
}

export async function readObjectFromFile(fileUrl) {
  const response = await fetch(fileUrl);
  const blob = await response.blob();
  const blobObject = JSON.parse(await blob.text());
  return blobObject;
}

function tensorDimsFromProto(dims) {
  // get rid of Long type for dims
  return dims.map(d => Long.isLong(d) ? d.toNumber() : d);
}

function tensorDataTypeFromProto(typeProto) {
  switch (typeProto) {
    case onnx.TensorProto.DataType.INT8:
      return 'int8';
    case onnx.TensorProto.DataType.UINT8:
      return 'uint8';
    case onnx.TensorProto.DataType.BOOL:
      return 'bool';
    case onnx.TensorProto.DataType.INT16:
      return 'int16';
    case onnx.TensorProto.DataType.UINT16:
      return 'uint16';
    case onnx.TensorProto.DataType.INT32:
      return 'int32';
    case onnx.TensorProto.DataType.UINT32:
      return 'uint32';
    case onnx.TensorProto.DataType.FLOAT:
      return 'float32';
    case onnx.TensorProto.DataType.DOUBLE:
      return 'float64';
    case onnx.TensorProto.DataType.STRING:
      return 'string';

    // For INT64/UINT64, reduce their value to 32-bits.
    // Should throw exception when overflow
    case onnx.TensorProto.DataType.INT64:
      return 'int64';
    case onnx.TensorProto.DataType.UINT64:
      return 'uint64';

    default:
      throw new Error(
          `unsupported data type: ${onnx.TensorProto.DataType[typeProto]}`);
  }
}

export async function downloadWeights(arg) {
  const response = await fetch(arg);
  const buf = await response.arrayBuffer();
  const modelProto = onnx.ModelProto.decode(new Uint8Array(buf));
  for (const i of modelProto.graph.initializer) {
    const tensor = {
      'data': Array.from(ort.JsTensor.Tensor.fromProto(i).data),
      'dims': tensorDimsFromProto(i.dims),
      'type': tensorDataTypeFromProto(i.dataType),
    };
    console.log(i.name + ',' + tensorDataTypeFromProto(i.dataType));
    const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
    writeObjectToFile(tensor, regName + '.json');
  }
}

export async function addWeights(map, arg) {
  const response = await fetch(arg);
  const buf = await response.arrayBuffer();
  const modelProto = onnx.ModelProto.decode(new Uint8Array(buf));
  for (const i of modelProto.graph.initializer) {
    const tensor = {
      'data': Array.from(ort.JsTensor.Tensor.fromProto(i).data),
      'dims': tensorDimsFromProto(i.dims),
      'type': tensorDataTypeFromProto(i.dataType),
    };
    console.log(i.name + ',' + tensorDataTypeFromProto(i.dataType));
    const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
    // writeObjectToFile(tensor, regName);
    map.set(name, tensor);
    ;
  }
}

export async function getOptimizedModel(modelName, save = false) {
  console.log('Dump - Optimize model begin.');
  const modelDir = './ort-models/';
  // const modelName = 'albert-base-v2';
  const graphOptimizationLevel = 'all';
  const optmizedModelName = modelName + '-' + graphOptimizationLevel + '.onnx';
  const optimizedModelFilePath = modelDir + optmizedModelName;
  let session;

  try {
    // create a new session and load the specific model.
    //
    // the model in this example contains a single MatMul node
    // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
    // it has 1 output: 'c'(float32, 3x3)
    const option = {
      executionProviders: [
        {
          name: 'wasm',
        },
      ],
      graphOptimizationLevel: graphOptimizationLevel,
      optimizedModelFilePath: optimizedModelFilePath,
    };
    session = await ort.InferenceSession.create(
        modelDir + modelName + '.onnx', option);
    console.log('Dump - Optimize model end.');

  } catch (e) {
    console.error(`failed to inference ONNX model: ${e}.`);
  }

  console.log(window.optmizedModelBlobUrl);
  const response = await fetch(window.optmizedModelBlobUrl);
  const blob = await response.blob();
  const arr = new Uint8Array(await blob.arrayBuffer());
  await session.release();
  return arr;
}

export class OnnxDumpData {
  constructor(modelName) {
    this.dumpDataMap = new Map();
    this.modelName = modelName;
  }

  async addWeights(optimizedModelBuffer) {
    // const response = await fetch(modelUrl);
    // const buf = await response.arrayBuffer();
    const modelProto = onnx.ModelProto.decode(optimizedModelBuffer);
    for (const i of modelProto.graph.initializer) {
      const tensor = {
        'data': Array.from(ort.JsTensor.Tensor.fromProto(i).data),
        'dims': tensorDimsFromProto(i.dims),
        'type': tensorDataTypeFromProto(i.dataType),
      };
      const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
      // writeObjectToFile(tensor, regName);
      this.dumpDataMap.set(regName, tensor);
      ;
    }
  }

  // Get the input put data.
  async addInputOutput(blobUrlMap) {
    for (const [key, value] of blobUrlMap.entries()) {
      // let response = await fetch(value);
      // const blob = await response.blob();
      // const blobObject = JSON.parse(await blob.text());
      console.log('In js: ' + key);
      const blobObject = await readObjectFromFile(value);
      // const arr = new Uint8Array(await blob.arrayBuffer());
      this.dumpDataMap.set(key, blobObject);
    }
  }

  save() {
    writeObjectToFile(this.dumpDataMap, this.modelName + '-inputoutput.json');
    return this.modelName + '-inputoutput';
  }

  getDumpData() {
    return this.dumpDataMap;
  }
}

export async function readObjectFromJson2(fileUrl) {
  let response = await fetch(fileUrl);
  const blob = await response.blob();
  const arr = new Uint8Array(await blob.arrayBuffer());
  return arr;
}

const onnxProto = ort.OnnxProto.onnx;

async function getDataFromJsonFile(modelName, fileUrl) {
  const deviceName = '';  //'-9106';  //'-7779';
                          // TODO: Fix constant node file not found.
  const response =
      await fetch(`./modeldata/${modelName}${deviceName}/` + fileUrl + '.json');
  const json = await response.json();
  if (json.type === 'float') {
    json.type = 'float32';
  }
  return json;
}

BigInt.prototype.toJSON = function() {
  return Number(this.toString());
};

function getParam(name) {
  name = name.replace(/[\[\]]/g, '\\$&');
  let regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)', 'i');
  let results = regex.exec(window.location.href);
  if (!results) return null;
  if (!results[2]) return '';
  return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

async function getData(inputName, node, dumpDataMap, modelName) {
  let data;
  const isMap = dumpDataMap instanceof Map;
  const regName = inputName.replace(/\//g, '_').replace(/:/g, '_');
  try {
    data = await getDataFromJsonFile(modelName, regName);
  } catch (err) {
    data = isMap ? dumpDataMap.get(regName) : dumpDataMap[regName];
  } finally {
    if (data == null) {
      console.error(
          ('Can not find input or output: ' + node.name + ', ' +
           JSON.stringify(node)));
      return null;
    }
    if (data.type === 'float') {
      data.type = 'float32';
    }
  }
  return data;
}

// opset is array: [{domain: '', version: 8}, ]
function getOpset(opType, opsets) {
  let opset = {'domain': 'com.microsoft', 'version': 1};
  if (opType === 'Add' || opType === 'Conv' || opType === 'Shape' ||
      opType === 'Reshape' || opType === 'Gather' || opType === 'Unsqueeze' ||
      opType === 'Concat' || opType === 'GlobalAveragePool' ||
      opType === 'Slice' || opType === 'Cast' || opType === 'Softmax' ||
      opType === 'MatMul' || opType === 'Sub' || opType === 'Mul' ||
      opType === 'Add' || opType === 'Div' || opType === 'LayerNormalization' ||
      opType === 'Transpose' || opType === 'Gemm' || opType === 'LeakyRelu' ||
      opType === 'MaxPool' || opType === 'BatchNormalization') {
    // {'domain': '', 'version': 12};
    // BatchNormalization, opset = {'domain': '', 'version': 8};
    opset.domain = '';
  }

  let versionFound = false;
  for (const item of opsets) {
    if (opset.domain === item.domain) {
      opset.version = item.version;
      versionFound = true;
    }
  }
  if (!versionFound) {
    throw new Error(
        'Not find domain: ' + JSON.stringify(opset.domain) + ' in ' +
        JSON.stringify(opsets));
  }
  return opset;
}

async function generateGraphPlan(node, dumpDataMap, modelName, model) {
  const nodePlan = {name: node.name};
  nodePlan.inputs = [];
  nodePlan.outputs = [];
  const inputShapeDefinitions = [];
  console.log(modelName + ', dump data ismap: ' + (dumpDataMap instanceof Map));
  for (const inputName of node.inputNames) {
    const inputData = await getData(inputName, node, dumpDataMap, modelName);
    nodePlan.inputs.push(inputData);
  }

  for (const outputName of node.outputNames) {
    let outputData = await getData(outputName, node, dumpDataMap, modelName);
    nodePlan.outputs.push(outputData);
  }

  for (const input of nodePlan['inputs']) {
    inputShapeDefinitions.push((input['dims']));
  }
  const attributs = [];
  node.attributes._attributes.forEach((value, key) => {
    attributs.push({'name': key, 'data': value[0], 'type': value[1]});
  });

  const opset = getOpset(node.opType, model._opsets);
  console.log(
      node.opType + ', ' +
      'domain: ' + JSON.stringify(opset));
  const graphPlan = {
    'name': node.opType,
    'operator': node.opType,
    'attributes': attributs,
    'inputShapeDefinitions': inputShapeDefinitions,
    'cases': [
      nodePlan,
    ],
    'backend': getParam('ep') || 'webgpu',
    'opset': opset,
    // TODO: fix opsetImport
    // 'opsetImport': model._opsets,
    //"opset":
    //[{"domain":"","version":12},{"domain":"com.microsoft.nchwc","version":1},{"domain":"ai.onnx.ml","version":3},{"domain":"com.ms.internal.nhwc","version":19},{"domain":"ai.onnx.training","version":1},{"domain":"ai.onnx.preview.training","version":1},{"domain":"com.microsoft","version":1},{"domain":"com.microsoft.experimental","version":1},{"domain":"org.pytorch.aten","version":1}]
  };

  return graphPlan;
}

async function runGraphPlan(graphPlan) {
  // ort.env.debug = true
  // ort.env.logLevel = 'verbose';
  const case0 = graphPlan['cases'][0];
  // TODO: outputs maybe array.
  const session = (await onnxdecoder.createOnnxModel(graphPlan, onnxProto));
  const result = await onnxdecoder.runOnnxProtoOp(session, case0);
  return result;
}

export async function loadModel(arg, byteOffset, length) {
  // const model = new onnx.Model();
  const model = new ort.Model();
  if (typeof arg === 'string') {
    const isOrtFormat = arg.endsWith('.ort');
    if (typeof process !== 'undefined' && process.versions &&
        process.versions.node) {
      // node
      const buf = await readFile(arg);
      model.load(buf);
    } else {
      // browser
      const response = await fetch(arg);
      const buf = await response.arrayBuffer();
      model.load(new Uint8Array(buf));
    }
  } else if (!ArrayBuffer.isView(arg)) {
    // load model from ArrayBuffer
    const arr = new Uint8Array(arg, byteOffset || 0, length || arg.byteLength);
    // this.initialize(arr);
    model.load(arr);
  } else {
    model.load(arg);
  }
  return model;
}

function convertArrayToBigInt64Array(array) {
  const bigint64array = new BigInt64Array(array.length);
  for (var i = 0; i < array.length; i++) {
    bigint64array[i] = BigInt(array[i]);
  }
  return bigint64array;
}

function compareIgnoreType(reference, result) {
  const isResultInt64 = result instanceof BigInt64Array;
  const referenceInt64 =
      isResultInt64 ? convertArrayToBigInt64Array(reference) : reference;
  if (isResultInt64) {
    return (
        JSON.stringify(referenceInt64.sort()) ===
        JSON.stringify(result.sort()));
  }
  return compare(referenceInt64, Array.from(result));
}

async function compareSingleNode(node, dumpDataMap, modelName, model) {
  const graphPlan =
      await generateGraphPlan(node, dumpDataMap, modelName, model);
  if (graphPlan == null) {
    return;
  }
  console.log(JSON.stringify(graphPlan));
  const result1 = await runGraphPlan(graphPlan);
  let reference = graphPlan['cases'][0]['outputs'][0].data;
  const compareResult = compareIgnoreType(reference, result1.output_0.cpuData);
  const compareInfo = 'Wasm vs ' + graphPlan['backend'] +
      ', compare result=' + compareResult + ',' + graphPlan['name'] + ', ' +
      graphPlan['cases'][0]['name'];
  if (compareResult) {
    console.log(compareInfo);
  } else {
    console.log('Compare reference : ' + JSON.stringify(reference));
    console.log(
        'Compare result : ' +
        JSON.stringify(Array.from(result1.output_0.cpuData)));
    console.error(
        'Wasm vs ' + graphPlan['backend'] + ', compare result=' +
        compareResult + ', failed node: ' + graphPlan['name'] + ', ' +
        graphPlan['cases'][0]['name'] + ', inputShapeDefinitions = ' +
        JSON.stringify(graphPlan['inputShapeDefinitions']));
  }
  return [compareResult, compareInfo];
}

export async function compareModel(model, dumpDataMap, modelName) {
  const nodes = model.graph._nodes;
  let testNode = getParam('node');
  // "/albert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/MatMul"
  // testNode = 'Conv_4';
  if (testNode) {
    for (const node of nodes) {
      if (testNode && node.name === testNode) {
        await compareSingleNode(node, dumpDataMap, modelName, model);
        break;
      }
    }
  } else {
    const results = [];
    for (const node of nodes) {
      const [compareResult, compareInfo] =
          await compareSingleNode(node, dumpDataMap, modelName, model);
      results.push({'result': compareResult, 'info': compareInfo});
    }
    writeObjectToFile(results, modelName + '-results.json');
  }
}

// dump(1)
export async function dump(modelName, runTaskFn, dumpOrCmp) {
  // const saveToFile = true;
  // const enableDump = false;
  let dumpDataMap;
  let optimizedModelBuffer;
  const optimizedModelName = modelName + '-opt.json';
  const optimizedModelDataName = modelName + '-opt-data.json';
  // When dumpOrCmp: 0, dump and cmp not from file.
  // 1, dump data to file; 2, cmp based on file.
  const useFile = dumpOrCmp != 0;


  if (dumpOrCmp != 2) {
    dumpDataMap = new OnnxDumpData(modelName);
    // 1. Generate optimized onnx file.
    window.dump = 2;
    optimizedModelBuffer = await getOptimizedModel(modelName);
    window.dump = 0;
    if (useFile) {
      writeObjectToFile2(optimizedModelBuffer, optimizedModelName);
    }
    // 2. Generate weights data.
    console.log('Dump - Generate weights data.');
    const modelUrl = `ort-models/${modelName}.onnx`;
    await dumpDataMap.addWeights(optimizedModelBuffer);
    console.log('Dump - Generate input output data.');
    // 3, Generate other dump data: input, output.
    window.dump = 1;
    await runTaskFn('performance', 'wasm');
    window.dump = 0;
    if (window.dumpBlobUrlMap != null) {
      await dumpDataMap.addInputOutput(window.dumpBlobUrlMap);
    }
    console.log('Dump - End.');
    if (useFile) {
      // writeObjectToFile works on mobilenet, not on albert.
      if (modelName == 'mobilenetv2-12') {
        writeMapToFile(dumpDataMap.getDumpData(), optimizedModelDataName);
        // writeObjectToFile(dumpDataMap.getDumpData(), optimizedModelDataName);
      } else {
        // For albert , too big.
        writeMapToFile(dumpDataMap.getDumpData(), optimizedModelDataName);
      }
    }
    dumpDataMap = dumpDataMap.getDumpData();
  }

  // 4, cmp
  console.log('Compare - Begin.');
  if (dumpOrCmp != 1) {
    if (useFile) {
      console.log(optimizedModelName);
      optimizedModelBuffer = await readObjectFromJson2(optimizedModelName);
      console.log(optimizedModelBuffer);
      // when cmp only, this means the dump data is from seperated file.
      dumpDataMap = dumpOrCmp == 2 ?
          null :
          await readObjectFromFile(optimizedModelDataName);
    }
    const model = await loadModel(optimizedModelBuffer);
    console.log(model);
    await compareModel(model, dumpDataMap, modelName);
  }
  console.log('Compare - End.');
}
