//===- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std ---===//
//
// پیاده‌سازی تبدیل جزئی (Partial Lowering) دیالکت Toy به ترکیبی از
// حلقه‌های Affine، حافظه‌های MemRef و عملیات استاندارد حسابی (Arith و Func).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine توابع کمکی
//===----------------------------------------------------------------------===//

/// تبدیل RankedTensorType به نوع معادل آن در بافر فیزیکی حافظه: MemRefType
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// تخصیص حافظه (memref.alloc) در ابتدای بلاک و آزادسازی آن (memref.dealloc) در انتهای بلاک
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = memref::AllocOp::create(rewriter, loc, type);

  // اطمینان از قرار گرفتن تخصیص در ابتدای بلاک جاری
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // درج آزادسازی در انتهای بلاک جاری (چون توابع Toy انشعاب پیچیده ندارند)
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// تابعی که رفتار محاسباتی درون عمیق‌ترین سطح حلقه را توصیف می‌کند
using LoopIterationFn =
    function_ref<Value(OpBuilder &rewriter, ValueRange loopIvs)>;

/// ایجاد حلقه‌های تودرتوی چندبعدی Affine متناظر با ابعاد تانسور خروجی
static void lowerOpToLoops(Operation *op, PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // تخصیص و آزادسازی بافر خروجی
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // تولید حلقه‌های تودرتو (یک حلقه به ازای هر بعد)
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, ivs);
        affine::AffineStoreOp::create(nestedBuilder, loc, valueToStore, alloc,
                                      ivs);
      });

  // جایگزینی عملگر با تخصیص حافظه خروجی
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// الگوهای تبدیل: عملیات دودویی (AddOp و MulOp)
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public OpConversionPattern<BinaryOp> {
  using OpConversionPattern<BinaryOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<BinaryOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      // لود کردن المان‌های متناظر از دو عملوند ورودی
      auto loadedLhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getLhs(), loopIvs);
      auto loadedRhs =
          affine::AffineLoadOp::create(builder, loc, adaptor.getRhs(), loopIvs);

      // انجام محاسبه اسکالر (arith.addf یا arith.mulf)
      return LoweredBinaryOp::create(builder, loc, loadedLhs, loadedRhs);
    });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// الگوهای تبدیل: ثابت‌ها (ConstantOp)
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // پیش‌تولید اندیس‌های ثابت به منظور جلوگیری از تکرار کد
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, i));
    } else {
      constantIndices.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    // ذخیره بازگشتی تک‌تک مقادیر ثابت در بافر با affine.store
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      if (dimension == valueShape.size()) {
        affine::AffineStoreOp::create(
            rewriter, loc, arith::ConstantOp::create(rewriter, loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
        return;
      }

      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    storeElements(/*dimension=*/0);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// الگوهای تبدیل: تعریف تابع (FuncOp)
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // تنها تابع main تبدیل می‌شود، چون باقی توابع قبلاً Inlined شده‌اند
    if (op.getName() != "main")
      return failure();

    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // ساخت تابع استاندارد func.func با همان بدنه
    auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                           op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// الگوهای تبدیل: چاپ (PrintOp)
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // toy.print حذف نمی‌شود، تنها عملوند ورودی آن به MemRef ارتقا می‌یابد
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// الگوهای تبدیل: بازگشت (ReturnOp)
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.hasOperand())
      return failure();

    // تبدیل toy.return به func.return
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// الگوهای تبدیل: ترانهاده (TransposeOp)
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public OpConversionPattern<toy::TransposeOp> {
  using OpConversionPattern<toy::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, rewriter, [&](OpBuilder &builder, ValueRange loopIvs) {
      Value input = adaptor.getInput();

      // لود با اندیس‌های معکوس برای انجام عمل ترانهاده
      SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
      return affine::AffineLoadOp::create(builder, loc, input, reverseIvs);
    });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
  StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // ۱. تعریف هدف تبدیل (ConversionTarget)
  ConversionTarget target(getContext());

  // دیالکت‌های استاندارد و قانونی در سطح پایین
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // علامت‌گذاری دیالکت Toy به عنوان غیرقانونی به جز استثنای پویا برای toy.print
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // ۲. افزودن الگوهای تبدیل (RewritePatternSet)
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
      &getContext());

  // ۳. اجرای تبدیل جزئی (Partial Conversion)
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
