#ifndef MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
#define MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {

/// تزریق اعلان‌های تولیدشده توسط TableGen برای ShapeInferenceOpInterface
#include "toy/ShapeInferenceOpInterfaces.h.inc"

} // namespace toy
} // namespace mlir

#endif // MLIR_TUTORIAL_TOY_SHAPEINFERENCEINTERFACE_H_
