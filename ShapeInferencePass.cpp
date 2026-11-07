#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;
using namespace toy;

/// تزریق تعاریف اینترفیس استنتاج شکل
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {

/// پاس استنتاج شکل درون‌تابعی (Intra-procedural Shape Inference Pass)
/// این پاس با استفاده از الگوریتم Worklist تمام عملگرهای با ابعاد ناشناخته را
/// تا رسیدن به نقطه ثابت (Fix-point) پردازش و ابعاد آن‌ها را استنتاج می‌کند.
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  StringRef getArgument() const override { return "toy-shape-inference"; }
  StringRef getDescription() const override {
    return "Infer shapes of operations in the Toy dialect";
  }

  void runOnOperation() override {
    auto f = getOperation();

    // ۱. ساخت Worklist شامل عملگرهایی که خروجی با شکل دینامیک دارند
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // ۲. پردازش حلقه‌ای تا خالی شدن لیست یا رسیدن به نقطه ثابت
    while (!opWorklist.empty()) {
      // پیدا کردن عملگری که تمام عملوندهای ورودی آن استنتاج و قطعی شده‌اند
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // فراخوانی متد inferShapes از طریق OpInterface
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // اگر لیست خالی نشده باشد، یعنی عملگرهایی وجود دارند که قابل حل نبوده‌اند
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// بررسی اینکه آیا تمام عملوندهای ورودی از نوع RankedTensor هستند
  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return llvm::isa<RankedTensorType>(operandType);
    });
  }

  /// بررسی اینکه آیا عملگر خروجی با ابعاد ناشناخته (غیر رنک‌دار) تولید می‌کند
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !llvm::isa<RankedTensorType>(resultType);
    });
  }
};

} // namespace

/// سازنده‌ی پاس استنتاج شکل
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
