#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::toy;

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
// ۳. AddOp و MulOp: inferShapes
// -----------------------------------------------------------------------------
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }

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
