#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

// =============================================================================
// ۱. الگوی ساده: حذف Reshape تو در تو
// Reshape(Reshape(x)) -> Reshape(x)
// =============================================================================
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

// =============================================================================
// ۲. فراخوانی کد C++ درون الگو (Native Code Call)
// Reshape(Constant(x)) -> Constant(x')
// تبدیل مقدار ثابت به شکلی جدید در زمان کامپایل (Constant Folding)
// =============================================================================
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;

def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

// =============================================================================
// ۳. الگوی همراه با شرط و قید (Constraint)
// Reshape(x) -> x در صورتی که نوع ورودی و خروجی کاملاً یکسان باشند
// =============================================================================
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE
