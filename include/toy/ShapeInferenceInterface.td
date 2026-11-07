#ifndef SHAPE_INFERENCE_INTERFACE
#define SHAPE_INFERENCE_INTERFACE

include "mlir/IR/OpBase.td"

/// تعریف اینترفیس استنتاج شکل در MLIR با استفاده از TableGen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    اینترفیس اختصاصی زبان Toy برای استنتاج ابعاد تانسور خروجی عملگرها
    از روی ابعاد عملوندهای ورودی.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
