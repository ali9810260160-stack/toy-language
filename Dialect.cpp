#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::toy;

// -----------------------------------------------------------------------------
// اینترفیس درون‌ریزی توابع (ToyInlinerInterface)
// -----------------------------------------------------------------------------
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// تمام فراخوانی‌ها در دیالکت Toy قابل درون‌ریزی هستند
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// تمام عملگرهای درون دیالکت Toy قابل درون‌ریزی در یک ریجن هستند
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// تمام توابع درون دیالکت Toy قابل درون‌ریزی هستند
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  /// مدیریت ترمیناتور (return) در زمان اینلاین کردن با جایگزینی مقادیر خروجی
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// تبدیل عدم تطابق نوع در زمان اینلاینینگ با درج یک CastOp
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return CastOp::create(builder, conversionLoc, resultType, input);
  }
};

// تزریق بدنه کلاس دیالکت
#include "toy/Dialect.cpp.inc"

// -----------------------------------------------------------------------------
// ۱. ثبت عملیات‌ها در دیالکت
// -----------------------------------------------------------------------------
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}

// -----------------------------------------------------------------------------
// ۲. ConstantOp: build, verify, parse, print
// -----------------------------------------------------------------------------
void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

LogicalResult ConstantOp::verify() {
  auto type = getType();
  auto attrType = getValue().getType();
  if (type != attrType)
    return emitOpError("constant type must match attribute type");
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  DenseElementsAttr value;
  if (parser.parseAttribute(value, "value", result.attributes))
    return failure();
  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttributeWithoutType(getValueAttr());
  p << " : " << getType();
}

// -----------------------------------------------------------------------------
// ۳. AddOp و MulOp: parse, print, inferShapes
// -----------------------------------------------------------------------------
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }
mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}
void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}
void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

// -----------------------------------------------------------------------------
// ۴. CastOp: inferShapes و areCastCompatible
// -----------------------------------------------------------------------------
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
  TensorType output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  return !input.hasRank() || !output.hasRank() || input == output;
}

// -----------------------------------------------------------------------------
// ۵. GenericCallOp: build و پیاده‌سازی متدهای CallOpInterface
// -----------------------------------------------------------------------------
void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

// -----------------------------------------------------------------------------
// ۶. ReturnOp: verify
// -----------------------------------------------------------------------------
LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());
  if (getNumOperands() > 0) {
    if (function.getFunctionType().getNumResults() == 0)
      return emitOpError("expected no return value");
    if (getOperand(0).getType() != function.getFunctionType().getResult(0))
      return emitOpError("return type mismatch");
  } else {
    if (function.getFunctionType().getNumResults() != 0)
      return emitOpError("expected return value");
  }
  return success();
}

// -----------------------------------------------------------------------------
// ۷. TransposeOp: build, verify, inferShapes
// -----------------------------------------------------------------------------
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

void TransposeOp::inferShapes() {
  auto arrayTy = llvm::cast<RankedTensorType>(getInput().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return success();

  auto inputShape = inputType.getShape();
  auto resultShape = resultType.getShape();
  
  if (inputShape.size() != resultShape.size())
    return emitOpError("input and result must have the same rank");

  for (size_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] != resultShape[inputShape.size() - 1 - i])
      return emitOpError("result shape must be the reverse of input shape");
  }
  return success();
}

// -----------------------------------------------------------------------------
// ۸. FuncOp: build, parse, print
// -----------------------------------------------------------------------------
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// تزریق بدنه (تعاریف) عملیات‌ها
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
