import Long from 'https://cdn.jsdelivr.net/npm/long@5.2.3/index.js';
import * as flatbuffersModule from './flatbuffers/js/flatbuffers.mjs'

// TODO: Fix this import.
const flatbuffers = flatbuffersModule.flatbuffers;
class LongUtil {
  // This function is called to get a number from long type of data for attribute, dim, and ir version,
  // which values are signed integers.
  // To make it more generic, add an optional paramter to convert to a unsigned number.
  static longToNumber(n, unsigned) {
    if (Long.isLong(n)) {
      return n.toNumber();
    } else if (n instanceof flatbuffers.Long) {
      return Long.fromValue({low: n.low, high: n.high, unsigned: unsigned ?? false}).toNumber();
    }
    return n;
  }
  static isLong(n) {
    return Long.isLong(n) || n instanceof flatbuffers.Long;
  }
}

export async  function createOnnxModel(test, onnx) {
  // const onnx = onnx;
  const opsetImport = onnx.OperatorSetIdProto.create(test.opset);

  const operator = test.operator;
  const attribute = (test.attributes || []).map(attr => {
    const protoAttr = onnx.AttributeProto.create({name: attr.name});
    switch (attr.type) {
      case 'float':
        protoAttr.type = onnx.AttributeProto.AttributeType.FLOAT;
        protoAttr.f = attr.data;
        break;
      case 'int':
        protoAttr.type = onnx.AttributeProto.AttributeType.INT;
        protoAttr.i = attr.data;
        break;
      case 'string':
        protoAttr.type = onnx.AttributeProto.AttributeType.STRING;
        protoAttr.s = new TextEncoder().encode(attr.data);
        break;
      case 'floats':
        protoAttr.type = onnx.AttributeProto.AttributeType.FLOATS;
        protoAttr.floats = attr.data;
        break;
      case 'ints':
        protoAttr.type = onnx.AttributeProto.AttributeType.INTS;
        protoAttr.ints = attr.data;
        break;
      case 'strings':
        protoAttr.type = onnx.AttributeProto.AttributeType.STRINGS;
        protoAttr.strings = (attr.data).map(s => new TextEncoder().encode(s));
        break;
      default:
        throw new Error(`Unsupported attribute type: ${attr.type}`);
    }
    return protoAttr;
  });

  if (test.cases.length === 0) {
    throw new Error(`No test cases found for test: ${test.name} [${test.operator}]`);
  }
  const inputCount = test.cases[0].inputs.length;
  const outputCount = test.cases[0].outputs.length;
  if (test.cases.some(
          testCase => testCase.inputs.length !== inputCount || testCase.outputs.length !== outputCount)) {
    throw new Error(
        `Test cases for test: ${test.name} [${test.operator}] must have the same number of inputs and outputs`);
  }

  const model = onnx.ModelProto.create();
  model.irVersion = onnx.Version.IR_VERSION;
  // TODO: fix opsetImport
  // model.opsetImport = test.opsetImport.map(i => ({domain: i.domain, version: LongUtil.longToNumber(i.version)}));
  console.log(model.opsetImport);
  model.opsetImport.push(opsetImport);
  //model.opsetImport.push({"domain":"com.microsoft","version":1});
  /*
  [{"domain":"","version":12},
                        {"domain":"com.microsoft.nchwc","version":1},{"domain":"ai.onnx.ml","version":3},
                        {"domain":"com.ms.internal.nhwc","version":19},{"domain":"ai.onnx.training","version":1},
                        {"domain":"ai.onnx.preview.training","version":1},
                        {"domain":"com.microsoft","version":1},{"domain":"com.microsoft.experimental","version":1},{"domain":"org.pytorch.aten","version":1}];
                      */
  model.graph = onnx.GraphProto.create();

  model.graph.node = [onnx.NodeProto.create({
    input: test.cases[0].inputs.map((_, i) => `input_${i}`),
    output: test.cases[0].outputs.map((_, i) => `output_${i}`),
    opType: operator,
    // TODO: fix opsetImport
    domain: test.opset?.domain, //{"domain":"com.microsoft","version":1},//
    name: operator,
    attribute
  })];

    // normalize input shape definitions
    let normalizedInputShapeDefinitions;
    if (!test.inputShapeDefinitions || test.inputShapeDefinitions === 'none') {
      // if inputShapeDefinitions is not specified, use undefined for all inputs
      normalizedInputShapeDefinitions = new Array(inputCount).fill(undefined);
    } else if (test.inputShapeDefinitions === 'rankOnly') {
      // check if all test cases have data
      if (test.cases.some(testCase => testCase.inputs.some(input => !input.data || !input.dims))) {
        throw new Error(`Test cases for test: ${test.name} [${
            test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`);
      }

      // if inputShapeDefinitions is 'rankOnly', use semantic names for all inputs. This means only rank is specified.
      normalizedInputShapeDefinitions =
          test.cases[0].inputs.map((input, i) => input.dims.map((_, j) => `_input_${i}_d${j}`));

      // check if all test cases have the same rank for each inputs
      if (test.cases.some(
              testCase => testCase.inputs.some(
                  (input, i) =>
                      input.dims.length !== (test.cases[0].inputs[i]).dims.length))) {
        throw new Error(`Test cases for test: ${test.name} [${
            test.operator}] must have the same rank for each inputs in different test cases`);
      }
    } else if (test.inputShapeDefinitions === 'static') {
      // check if all test cases have data
      if (test.cases.some(testCase => testCase.inputs.some(input => !input.data || !input.dims))) {
        throw new Error(`Test cases for test: ${test.name} [${
            test.operator}] must have data for each inputs when inputShapeDefinitions is 'rankOnly'`);
      }

      // if inputShapeDefinitions is 'static', use the shape of the first test case for all inputs.
      normalizedInputShapeDefinitions = test.cases[0].inputs.map((input) => input.dims);

      // check if all test cases have the same shape for each inputs
      if (test.cases.some(
              testCase => testCase.inputs.some(
                  (input, i) => TensorResultValidator.integerEqual(
                      input.dims, (test.cases[0].inputs[i]).dims)))) {
        throw new Error(`Test cases for test: ${test.name} [${
            test.operator}] must have the same shape for each inputs in different test cases`);
      }
    } else {
      // if inputShapeDefinitions is specified as an array, use it as is.
      // check if inputShapeDefinitions has the same number of inputs as test cases
      if (test.inputShapeDefinitions && test.inputShapeDefinitions.length !== inputCount) {
        throw new Error(
            `Input shape definitions for test: ${test.name} [${test.operator}] must have the same number of inputs`);
      }
      normalizedInputShapeDefinitions = test.inputShapeDefinitions;
    }

    model.graph.input = test.cases[0].inputs.map((input, i) => {
      const shapeDefinition = normalizedInputShapeDefinitions[i];
      const shape = shapeDefinition ? onnx.TensorShapeProto.create({
        dim: shapeDefinition.map(
            dim => onnx.TensorShapeProto.Dimension.create(typeof dim === 'string' ? {dimParam: dim} : {dimValue: dim}))
      }) :
                                      undefined;
      return onnx.ValueInfoProto.create({
        name: `input_${i}`,
        type: onnx.TypeProto.create({
          tensorType: onnx.TypeProto.Tensor.create({elemType: tensorDataTypeStringToEnum(input.type), shape}),
        }),
      });
    });

    model.graph.output = test.cases[0].outputs.map((output, i) => onnx.ValueInfoProto.create({
      name: `output_${i}`,
      type: onnx.TypeProto.create({
        tensorType: onnx.TypeProto.Tensor.create({elemType: tensorDataTypeStringToEnum(output.type)}),
      }),
    }));

    model.graph.name = test.name;

    const backendHint = test.backend;
    const loadedData = onnx.ModelProto.encode(model).finish();

    const session = await ort.InferenceSession.create(
      loadedData, {executionProviders: [backendHint]});
  return  session;
}

export async function runOnnxProtoOp(
  session, testCase){
const feeds = {};//: Record<string, ort.Tensor> = {};
const fetches = [];//: string[] = [];
testCase.inputs.forEach((input, i) => {
  if (input.data) {
    let data = input.data;
    if (input.type === 'uint64') {
      data = BigUint64Array.from(input.data.map(BigInt));
    } else if (input.type === 'int64') {
      data = BigInt64Array.from(input.data.map(BigInt));
    }
    feeds[`input_${i}`] = new ort.Tensor(input.type, data, input.dims);
  }
});

const outputs = [];//: ort.Tensor[] = [];
const expectedOutputNames = [];// string[] = [];
testCase.outputs.forEach((output, i) => {
  if (output.data) {
    let data = output.data;
    if (output.type === 'uint64') {
      data = BigUint64Array.from(output.data.map(BigInt));
    } else if (output.type === 'int64') {
      data = BigInt64Array.from(output.data.map(BigInt));
    }
    outputs.push(new ort.Tensor(output.type, data, output.dims));
    expectedOutputNames.push(`output_${i}`);
    fetches.push(`output_${i}`);
  }
});

const results = await session.run(feeds, fetches);
return results;

// const actualOutputNames = Object.getOwnPropertyNames(results);
// expect(actualOutputNames.length).to.equal(expectedOutputNames.length);
// expect(actualOutputNames).to.have.members(expectedOutputNames);
// 
// const actualOutputs = actualOutputNames.map(name => results[name]);
// validator.checkApiTensorResult(actualOutputs, outputs);
}


const tensorDataTypeStringToEnum = (type) => {
  switch (type) {
      case 'int8':
          return 3 /* DataType.int8 */;
      case 'uint8':
          return 2 /* DataType.uint8 */;
      case 'bool':
          return 9 /* DataType.bool */;
      case 'int16':
          return 5 /* DataType.int16 */;
      case 'uint16':
          return 4 /* DataType.uint16 */;
      case 'int32':
          return 6 /* DataType.int32 */;
      case 'uint32':
          return 12 /* DataType.uint32 */;
      case 'float16':
          return 10 /* DataType.float16 */;
      case 'float32':
          return 1 /* DataType.float */;
      case 'float64':
          return 11 /* DataType.double */;
      case 'string':
          return 8 /* DataType.string */;
      case 'int64':
          return 7 /* DataType.int64 */;
      case 'uint64':
          return 13 /* DataType.uint64 */;
      default:
          throw new Error(`unsupported data type: ${type}`);
  }
};
