#include "toy/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
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
// -----------------------------------------------------------------------------
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
// ۳. ReturnOp: verify
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
// ۴. TransposeOp: verify
// -----------------------------------------------------------------------------
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
// ۵. FuncOp: build, parse, print
// -----------------------------------------------------------------------------
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (type.getNumInputs() == 0)
    return;

  Block *entryBlock = new Block();
  state.regions.front()->push_back(entryBlock);
  for (Type t : type.getInputs())
    entryBlock->addArgument(t, state.location);
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