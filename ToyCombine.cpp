#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"

using namespace mlir;
using namespace toy;

/// الگوی بازنویسی به زبان C++ برای TransposeOp
/// این الگو سناریوی زیر را بهینه‌سازی و ساده‌سازی می‌کند:
/// transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// با اولویت (Benefit) برابر ۱ الگو را ثبت می‌کنیم
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// این متد الگو را بررسی کرده و در صورت انطباق، IR را بازنویسی می‌کند
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // ۱. بررسی عملوند ورودی ترنسپوز جاری
    mlir::Value transposeInput = op.getOperand();

    // ۲. بررسی اینکه آیا این مقدار حاصل یک TransposeOp دیگر است؟
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // اگر نبود، این الگو مطابقت ندارد و شکست بازمی‌گرداند
    if (!transposeInputOp)
      return failure();

    // ۳. در غیر این صورت، ترنسپوز دوم زائد است؛
    // عملوند ورودیِ ترنسپوز اول را مستقیماً جایگزین این عملیات می‌کنیم
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// ثبت الگوهای کانونیکالایزیشن برای TransposeOp
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
