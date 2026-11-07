#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
/// اعلان سازنده‌ی پاس استنتاج شکل (Shape Inference Pass)
std::unique_ptr<Pass> createShapeInferencePass();

/// پاس تبدیل عملیات Toy به حلقه‌های Affine و تخصیص‌های MemRef (فصل ۵)
std::unique_ptr<Pass> createLowerToAffinePass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
