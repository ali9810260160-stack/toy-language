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

/// پاس تبدیل عملیات باقی‌مانده (toy.print و استانداردها) به دیالکت LLVM (فصل ۶)
std::unique_ptr<Pass> createLowerToLLVMPass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
