import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const runtimeDirIndex = process.argv.indexOf("--runtime-dir");
const runtimeDirectory = runtimeDirIndex === -1
  ? null
  : resolve(process.argv[runtimeDirIndex + 1] || "");

const sourceBundle = new URL(
  "../node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs",
  import.meta.url,
);
const sourceWasm = new URL(
  "../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
  import.meta.url,
);
const vendorDirectory = new URL("../src/vendor/", import.meta.url);
const modelDirectory = new URL("../src/model/", import.meta.url);
const workerBundle = new URL("ort.wasm.cloudflare.mjs", vendorDirectory);
const workerWasm = new URL("ort-wasm-simd-threaded.wasm", modelDirectory);

const binaryConversion = "if(R==K&&g)R=new Uint8Array(g);";
const moduleAwareConversion =
  "if(R==K&&g)R=g instanceof WebAssembly.Module?g:new Uint8Array(g);";
const instantiateCall = "return await WebAssembly.instantiate(x,S)";
const normalizedInstantiate =
  "let cf=await WebAssembly.instantiate(x,S);return x instanceof WebAssembly.Module?{instance:cf,module:x}:cf";

await mkdir(vendorDirectory, { recursive: true });
await mkdir(modelDirectory, { recursive: true });

if (runtimeDirectory) {
  await copyFile(
    resolve(runtimeDirectory, "ort.wasm.cloudflare.mjs"),
    workerBundle,
  );
  await copyFile(
    resolve(runtimeDirectory, "ort-wasm-simd.wasm"),
    workerWasm,
  );
  await copyFile(
    resolve(runtimeDirectory, "runtime_manifest.json"),
    new URL("runtime_manifest.json", modelDirectory),
  );
  console.log("Installed the operator-reduced ONNX Runtime build.");
  process.exit(0);
}

const source = await readFile(sourceBundle, "utf8");
const conversionOccurrences = source.split(binaryConversion).length - 1;
const instantiateOccurrences = source.split(instantiateCall).length - 1;
if (conversionOccurrences !== 1 || instantiateOccurrences !== 1) {
  throw new Error(
    "Expected one ONNX Runtime wasmBinary conversion and instantiation site; " +
      `found ${conversionOccurrences} and ${instantiateOccurrences}. ` +
      "The pinned runtime adapter needs review before deployment.",
  );
}

const adapted = source
  .replace(binaryConversion, moduleAwareConversion)
  .replace(instantiateCall, normalizedInstantiate);
await writeFile(workerBundle, adapted);
await copyFile(sourceWasm, workerWasm);

console.log("Prepared ONNX Runtime for Cloudflare's precompiled WASM modules.");
