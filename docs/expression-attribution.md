# Portrait expression model attribution

This feature uses ONNX exports of the mesh and blendshape models from the Google
MediaPipe Face Landmarker bundle, version `face_landmarker/float16/1`. The models
and the 146-landmark subset/52-label order are Apache-2.0, MediaPipe Authors/Google.
The application does not depend on the MediaPipe Python package.

- [MediaPipe source and license](https://github.com/google-ai-edge/mediapipe)
- [Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)
- [Official Face Landmarker documentation and model cards](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [Versioned model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- [Pinned ONNX exports, conversion script and numerical parity notes](https://huggingface.co/thedemons/mirayu-models/tree/dc44090caa7ce49c81139e4fd7c22b63f81943a3/mediapipe-face-landmarker)
- [Upstream blendshape input subset and label ordering](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc)

The exports were produced by the Mirayu project using tf2onnx opset 17 with the
original NHWC layout and float16 weights promoted to float32. Their model card
records the original Google bundle/TFLite hashes and conversion parity checks.
The application downloads pinned exports with exact size/SHA-256 checks;
it does not modify weights or require a user-side model conversion. Its explicit
neutral-expression scoring heuristic combines visible blendshape activity;
it is not an upstream neutral-expression classifier.
