import Long from "https://cdn.jsdelivr.net/npm/long@5.2.3/index.js";

const onnx = ort.OnnxProto.onnx; //onnxModule.onnx;

function sleep(ms) {
  let start = new Date().getTime();
  let expire = start + ms;
  while (new Date().getTime() < expire) {
  }
  return;
}

export function saveObjectsToFile(json_object, name) {
  // const name = json_object['name'];
  const object = json_object;
  const file_name = `${name}.json`;
  const a = document.createElement('a');
  const file = new Blob([JSON.stringify(object)], {type: 'application/json'});
  a.href = URL.createObjectURL(file);
  a.download = file_name;
  a.click();
  sleep(100);
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
    console.log(i.name + "," + tensorDataTypeFromProto(i.dataType));
    const regName = i.name.replace(/\//g, '_').replace(/:/g, '_');
    await saveObjectsToFile(tensor, regName)
  }
}
