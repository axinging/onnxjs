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

export function writeObjectToFile(jsonObject, name, time = 200) {
  let object = jsonObject;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  let jsonStr = name.split('.').pop() === 'jsonc' ? JSON.stringify([object]) :
                                                    JSON.stringify(object);
  const file = new Blob([jsonStr], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
  sleep(time);
}

export function writeMapToFile(jsonObject, name, time = 200) {
  let object = jsonObject;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    for (let [key, value] of object) {
      // console.log(key + ' = ' + value);
      writeObjectToFile(value, key + '.json');
    }
  }
}

export function writeTypedArrayToFile(tyepdArray, name, time = 200) {
  let object = tyepdArray;
  const fileName = name;
  const a = document.createElement('a');
  if (object instanceof Map) {
    object = Object.fromEntries(object);
  }
  const file = new Blob([object], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
  sleep(time);
}

export async function readTypedArrayFromFile(fileUrl) {
  let response = await fetch(fileUrl);
  const blob = await response.blob();
  const tyepdArray = new Uint8Array(await blob.arrayBuffer());
  return tyepdArray;
}

async function readFromJsonFile(fileUrl) {
  const response = await fetch(fileUrl);
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

async function runGraphPlan(graphPlan) {
  // ort.env.debug = true
  // ort.env.logLevel = 'verbose';

  // TODO: outputs maybe array.
  const session = await onnxdecoder.createOnnxModel(graphPlan, onnx);
  const result = await onnxdecoder.runOnnxProtoOp(graphPlan, session);
  return result;
}

export async function loadModel(arg) {
  // ort.Model().
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

function getDirInfo(modelName, graphOptimizationLevel) {
  const optimizedModelName = modelName + '-' + graphOptimizationLevel + '.json';
  const optimizedModelDataName =
      modelName + '-' + graphOptimizationLevel + '-data.json';
  const modelDir =
      './modeldata/' + modelName + '-' + graphOptimizationLevel + '/';
  return [modelDir, optimizedModelName, optimizedModelDataName];
}

export class OnnxDumpData {
  constructor(modelName, graphOptimizationLevel, dumpOrCmp) {
    this.dumpDataMap = new Map();
    this.optimizedModelBuffer = null;
    this.graphOptimizationLevel = graphOptimizationLevel ?? 'all';
    this.dumpOrCmp = Number(dumpOrCmp);
    this.useFile = dumpOrCmp != 0;
    this.modelName = modelName;

    const [modelDir, optimizedModelName, optimizedModelDataName] =
        getDirInfo(modelName, graphOptimizationLevel);
    Object.assign(this, {modelDir, optimizedModelName, optimizedModelDataName});

    this.model = null;
  }

  release() {
    // TODO: 
  }

  async setupWeights() {
    const modelProto = onnx.ModelProto.decode(this.optimizedModelBuffer);
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
  async setupInputOutputs() {
    if (window.dumpBlobUrlMap == null) {
      throw new Error('window.dumpBlobUrlMap is NULL!');
    }
    const blobUrlMap = window.dumpBlobUrlMap;
    for (const [key, value] of blobUrlMap.entries()) {
      const blobObject = await readObjectFromFile(value);
      // const arr = new Uint8Array(await blob.arrayBuffer());
      this.dumpDataMap.set(key, blobObject);
    }
  }

  async setup(runTaskFn) {
    window.dump = 2;
    const optimizedModelBuffer = await this.getOptimizedModel();
    const optimizedModelName = this.optimizedModelName;
    window.dump = 0;
    if (this.useFile) {
      writeTypedArrayToFile(optimizedModelBuffer, optimizedModelName);
    }
    // 2. Generate weights data.
    console.log('Dump - Generate weights data.');
    await this.setupWeights(optimizedModelBuffer);
    console.log('Dump - Generate input output data.');
    // 3, Generate other dump data: input, output.
    window.dump = 1;
    await runTaskFn('performance', 'wasm');
    window.dump = 0;
    await this.setupInputOutputs();
  }

  async getOptimizedModel() {
    const modelName = this.modelName;
    const modelDir = this.modelDir;
    console.log('Dump - Optimize model begin.');
    const graphOptimizationLevel = this.graphOptimizationLevel;
    const optimizedModelFilePath = modelName + '-' + graphOptimizationLevel + '.onnx';
    let session;

    try {
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
      console.error(`Failed to inference ONNX model: ${e}.`);
    }

    console.log(window.optmizedModelBlobUrl);
    const response = await fetch(window.optmizedModelBlobUrl);
    const blob = await response.blob();
    this.optimizedModelBuffer = new Uint8Array(await blob.arrayBuffer());
    // await session.release();
    return this.optimizedModelBuffer;
  }

  save() {
    // writeObjectToFile(this.dumpDataMap, this.modelName +
    // '-inputoutput.json'); return this.modelName + '-inputoutput';
    const optimizedModelDataName = this.optimizedModelDataName;
    const modelName = this.modelName;
    const dumpDataMap = this.dumpDataMap;
    if (this.useFile) {
      // writeObjectToFile works on mobilenet, not on albert.
      if (modelName == 'mobilenetv2-12') {
        writeMapToFile(dumpDataMap, optimizedModelDataName);
        // writeObjectToFile(dumpDataMap.getDumpData(), optimizedModelDataName);
      } else {
        // For albert, too big.
        writeMapToFile(dumpDataMap, optimizedModelDataName);
      }
    }
  }

  async restore() {
    if (this.useFile) {
      console.log(this.optimizedModelName);
      if (this.optimizedModelBuffer == null) {
        this.optimizedModelBuffer = await readTypedArrayFromFile(
            this.modelDir + this.optimizedModelName);
      }
      console.log(this.optimizedModelBuffer);
      // when cmp only, this means the dump data is from seperated file.
      this.dumpDataMap = this.dumpOrCmp == 2 ?
          null :
          await readObjectFromFile(this.modelDir + this.optimizedModelDataName);
    }
  }

  async compare() {
    this.model = await loadModel(this.optimizedModelBuffer);
    await this.compareModel();
  }

  async setupGraphPlan(node) {
    const dumpDataMap = this.dumpDataMap;
    const modelName = this.modelName;
    const model = this.model;

    const nodePlan = {name: node.name};
    nodePlan.inputs = [];
    nodePlan.outputs = [];
    const inputShapeDefinitions = [];
    console.log(
        modelName + ', dump data ismap: ' + (dumpDataMap instanceof Map));
    for (const inputName of node.inputNames) {
      const inputData = await this.getData(inputName, node);
      nodePlan.inputs.push(inputData);
    }

    for (const outputName of node.outputNames) {
      let outputData = await this.getData(outputName, node);
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
    };

    return graphPlan;
  }

  async getData(inputName, node) {
    const dumpDataMap = this.dumpDataMap;
    let data;
    const isMap = dumpDataMap instanceof Map;
    const regName = inputName.replace(/\//g, '_').replace(/:/g, '_');
    try {
      data = await readFromJsonFile(this.modelDir + regName  + '.json');
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

  async compareSingleNode(node) {
    const graphPlan = await this.setupGraphPlan(node);
    if (graphPlan == null) {
      return;
    }
    const result1 = await runGraphPlan(graphPlan);
    let reference = graphPlan['cases'][0]['outputs'][0].data;
    const compareResult =
        compareIgnoreType(reference, result1.output_0.cpuData);
    const compareInfo = 'Wasm vs ' + graphPlan['backend'] +
        ', compare result=' + compareResult + ',' + graphPlan['name'] + ', ' +
        graphPlan['cases'][0]['name'];
    if (compareResult) {
      // console.log(compareInfo);
      // writeObjectToFile(graphPlan, graphPlan['cases'][0]['name'] + ".jsonc");
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
      writeObjectToFile(
          graphPlan,
          graphPlan['cases'][0]['name'] + '-' + this.graphOptimizationLevel +
              '.jsonc');
    }
    return [compareResult, compareInfo];
  }

  async compareModel() {
    const model = this.model;
    const dumpDataMap = this.dumpDataMap;
    const modelName = this.modelName;
    const nodes = model.graph._nodes;
    let testNode = getParam('node');
    if (testNode) {
      for (const node of nodes) {
        if (testNode && node.name === testNode) {
          await this.compareSingleNode(node);
          break;
        }
      }
    } else {
      const results = [];
      for (const node of nodes) {
        const [compareResult, compareInfo] =
            await this.compareSingleNode(node, dumpDataMap, modelName, model);
        results.push({'result': compareResult, 'info': compareInfo});
      }
      writeObjectToFile(results, modelName + '-results.json');
    }
  }

  getDumpData() {
    return this.dumpDataMap;
  }
}

async function getData(inputName, node, dumpDataMap, modelDir) {
  let data;
  const isMap = dumpDataMap instanceof Map;
  const regName = inputName.replace(/\//g, '_').replace(/:/g, '_');
  try {
    data = await readFromJsonFile(modelDir + regName + '.json');
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

export async function setupWeights(map, optimizedModelBuffer) {
  // const response = await fetch(arg);
  // const buf = await response.arrayBuffer();
  // const modelProto = onnx.ModelProto.decode(new Uint8Array(buf));

  const modelProto = onnx.ModelProto.decode(optimizedModelBuffer);
  for (const i of modelProto.graph.initializer) {
    const tensor = {
      'data': Array.from(ort.JsTensor.Tensor.fromProto(i).data),
      'dims': tensorDimsFromProto(i.dims),
      'type': tensorDataTypeFromProto(i.dataType),
    };
    const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
    map.set(regName, tensor);
  }
}

// Get the input put data.
export async function setupInputOutputs(dumpDataMap) {
  if (window.dumpBlobUrlMap == null) {
    throw new Error('window.dumpBlobUrlMap is NULL!');
  }
  const blobUrlMap = window.dumpBlobUrlMap;
  for (const [key, value] of blobUrlMap.entries()) {
    const blobObject = await readObjectFromFile(value);
    // const arr = new Uint8Array(await blob.arrayBuffer());
    dumpDataMap.set(key, blobObject);
  }
}

export async function getOptimizedModel(
    modelDir, modelName, graphOptimizationLevel) {
  console.log('Dump - Optimize model begin.');
  // const modelDir = './ort-models/';
  // const graphOptimizationLevel = 'all';
  const optimizedModelFilePath =
      modelDir + modelName + '-' + graphOptimizationLevel + '.onnx';
  ;
  let session;

  try {
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
    console.error(`Failed to inference ONNX model: ${e}.`);
  }

  console.log(window.optmizedModelBlobUrl);
  const response = await fetch(window.optmizedModelBlobUrl);
  const blob = await response.blob();
  const arr = new Uint8Array(await blob.arrayBuffer());
  // await session.release();
  return arr;
}

async function setupGraphPlan(node, dumpDataMap, modelDir, modelName, model) {
  const nodePlan = {name: node.name};
  nodePlan.inputs = [];
  nodePlan.outputs = [];
  const inputShapeDefinitions = [];
  console.log(modelName + ', dump data ismap: ' + (dumpDataMap instanceof Map));
  for (const inputName of node.inputNames) {
    const inputData = await getData(inputName, node, dumpDataMap, modelDir);
    nodePlan.inputs.push(inputData);
  }

  for (const outputName of node.outputNames) {
    let outputData = await getData(outputName, node, dumpDataMap, modelDir);
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
  };

  return graphPlan;
}


async function compareSingleNode(
    node, dumpDataMap, modelDir, modelName, model) {
  const graphPlan =
      await setupGraphPlan(node, dumpDataMap, modelDir, modelName, model);
  if (graphPlan == null) {
    return;
  }
  // console.log(JSON.stringify(graphPlan));
  const result1 = await runGraphPlan(graphPlan);
  let reference = graphPlan['cases'][0]['outputs'][0].data;
  const compareResult = compareIgnoreType(reference, result1.output_0.cpuData);
  const compareInfo = 'Wasm vs ' + graphPlan['backend'] +
      ', compare result=' + compareResult + ',' + graphPlan['name'] + ', ' +
      graphPlan['cases'][0]['name'];
  if (compareResult) {
    console.log(compareInfo);
    // writeObjectToFile(graphPlan, graphPlan['cases'][0]['name'] + ".jsonc");
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
    writeObjectToFile(graphPlan, graphPlan['cases'][0]['name'] + '.jsonc');
  }
  return [compareResult, compareInfo];
}

export async function compareModel(model, dumpDataMap, modelDir, modelName) {
  const nodes = model.graph._nodes;
  let testNode = getParam('node');
  // "/albert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/MatMul"
  // testNode = 'Conv_4';
  if (testNode) {
    for (const node of nodes) {
      if (testNode && node.name === testNode) {
        await compareSingleNode(node, dumpDataMap, modelDir, modelName, model);
        break;
      }
    }
  } else {
    const results = [];
    for (const node of nodes) {
      const [compareResult, compareInfo] = await compareSingleNode(
          node, dumpDataMap, modelDir, modelName, model);
      results.push({'result': compareResult, 'info': compareInfo});
    }
    writeObjectToFile(results, modelName + '-results.json');
  }
}


export async function dump(
    modelName, runTaskFn, graphOptimizationLevel = 'disabled',
    dumpOrCmp = '0') {
  // When dumpOrCmp: 0, dump and cmp not from file.
  // 1, dump data to file; 2, cmp based on file.
  const useFile = dumpOrCmp != 0;

  const useClass = getParam('useclass') == 'false' ? false : true;
  console.log("useclass = " + useClass);

  if (useClass) {
    const dumpDataMap =
        new OnnxDumpData(modelName, graphOptimizationLevel, dumpOrCmp);
    if (dumpOrCmp != 2) {
      await dumpDataMap.setup(runTaskFn);
      if (useFile) {
        dumpDataMap.save();
      }
    }
    if (dumpOrCmp != 1) {
      console.log('Compare - Begin.');
      if (useFile) {
        await dumpDataMap.restore();
      }
      await dumpDataMap.compare();
      dumpDataMap.release();
      console.log('Compare - End.');
    }
  } else {
    let dumpDataMap;
    let optimizedModelBuffer;
    // const optimizedModelName = modelName + '-opt.json';
    // const optimizedModelDataName = modelName + '-opt-data.json';
    graphOptimizationLevel = graphOptimizationLevel ?? 'disabled';
    // const optimizedModelName = modelName + '-'  + graphOptimizationLevel
    // +'.json'; const optimizedModelDataName = modelName + '-' +
    // graphOptimizationLevel + '-data.json'; const modelDir =
    // './modeldata/'+modelName + '-'  + graphOptimizationLevel + '/';

    const [modelDir, optimizedModelName, optimizedModelDataName] =
        getDirInfo(modelName, graphOptimizationLevel);

    dumpDataMap = new Map();
    if (dumpOrCmp != 2) {
      // 1. Generate optimized onnx file.
      window.dump = 2;
      optimizedModelBuffer =
          await getOptimizedModel(modelDir, modelName, graphOptimizationLevel);
      window.dump = 0;
      if (useFile) {
        writeTypedArrayToFile(optimizedModelBuffer, optimizedModelName);
      }
      // 2. Generate weights data.
      console.log('Dump - Generate weights data.');
      await setupWeights(dumpDataMap, optimizedModelBuffer);
      console.log('Dump - Generate input output data.');
      // 3, Generate other dump data: input, output.
      window.dump = 1;
      await runTaskFn('performance', 'wasm');
      window.dump = 0;
      await setupInputOutputs(dumpDataMap);
      console.log('Dump - End.');
      if (useFile) {
        // writeObjectToFile works on mobilenet, not on albert.
        if (modelName == 'mobilenetv2-12') {
          writeMapToFile(dumpDataMap, optimizedModelDataName);
          // writeObjectToFile(dumpDataMap.getDumpData(),
          // optimizedModelDataName);
        } else {
          // For albert, too big.
          writeMapToFile(dumpDataMap, optimizedModelDataName);
        }
      }
    }
    // 4, cmp
    console.log('Compare - Begin.');
    if (dumpOrCmp != 1) {
      if (useFile) {
        console.log(optimizedModelName);
        optimizedModelBuffer =
            await readTypedArrayFromFile(modelDir + optimizedModelName);
        console.log(optimizedModelBuffer);
        // when cmp only, this means the dump data is from seperated file.
        dumpDataMap = dumpOrCmp == 2 ?
            null :
            await readObjectFromFile(modelDir + optimizedModelDataName);
      }
      const model = await loadModel(optimizedModelBuffer);
      console.log(model);
      await compareModel(model, dumpDataMap, modelDir, modelName);
      console.log('Compare - End.');
    }
  }
}
