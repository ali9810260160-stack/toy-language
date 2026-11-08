//===- LowerToLLVM.cpp - Lowering from Toy+Affine+Std to LLVM -------------===//
//
// تبدیل کامل به دیالکت LLVM و کدهای میانی ماشین.
// عملیات toy.print به حلقه‌های تودرتوی scf و فراخوانی printf تبدیل می‌شود.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// الگوی تبدیل toy.print به حلقه‌های فراخوانی تابع خارجی printf
//===----------------------------------------------------------------------===//

namespace {
class PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
public:
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // ثبت یا دریافت اعلان تابع خارجی printf
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // ایجاد حلقه‌های استاندارد تودرتو (scf.for) برای پیمایش ابعاد
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto upperBound =
          arith::ConstantIndexOp::create(rewriter, loc, memRefShape[i]);
      auto step = arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto loop =
          scf::ForOp::create(rewriter, loc, lowerBound, upperBound, step);
      for (Operation &nested : make_early_inc_range(*loop.getBody()))
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // خاتمه دادن به بدنه حلقه
      rewriter.setInsertionPointToEnd(loop.getBody());

      // چاپ خط جدید پس از پایان هر بعد داخلی
      if (i != e - 1)
        LLVM::CallOp::create(rewriter, loc, getPrintfType(context), printfRef,
                             newLineCst);
      scf::YieldOp::create(rewriter, loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // لود کردن المان فعلی از بافر و فراخوانی printf
    auto elementLoad =
        memref::LoadOp::create(rewriter, loc, op.getInput(), loopIvs);
    LLVM::CallOp::create(rewriter, loc, getPrintfType(context), printfRef,
                         ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// تعریف امضای تابع printf: i32 (ptr, ...)
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);
    return llvmFnType;
  }

  /// تعریف تابع printf درون ModuleOp در صورت عدم وجود
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf",
                             getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }

  /// ایجاد رشته سراسری (Global String) مثل فرمت چاپ "%f "
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/true,
                                      LLVM::Linkage::Internal, name,
                                      builder.getStringAttr(value),
                                      /*alignment=*/0);
    }

    Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
    Value cst0 = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                          builder.getIndexAttr(0));
    return LLVM::GEPOp::create(
        builder, loc, LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// پاس تبدیل جامع به LLVM
//===----------------------------------------------------------------------===//

namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
  StringRef getArgument() const override { return "toy-to-llvm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  patterns.add<PrintOpLowering>(&getContext());

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
