#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"

// -----------------------------------------------------------------------------
// تعریف دیالکت Toy
// -----------------------------------------------------------------------------
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A tutorial Toy dialect";
  let description = [{
    The Toy dialect is a simple dialect designed to demonstrate the
    capabilities of MLIR. It defines a simple tensor-based language.
  }];
  let cppNamespace = "::mlir::toy";
}

// -----------------------------------------------------------------------------
// کلاس پایه برای تمام عملیات‌های Toy
// -----------------------------------------------------------------------------
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// =============================================================================
// ۱. ConstantOp: تعریف یک مقدار ثابت
// =============================================================================
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "Constant operation";
  let description = [{ Defines a constant tensor. }];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// =============================================================================
// ۲ و ۳. AddOp و MulOp
// =============================================================================
def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, UnrankedTensorType::get($_builder.getF64Type()),
            lhs, rhs);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Multiplication operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, UnrankedTensorType::get($_builder.getF64Type()),
            lhs, rhs);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

// =============================================================================
// ۴. CastOp: تبدیل شکل تانسور
// =============================================================================
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

// =============================================================================
// ۵. GenericCallOp: فراخوانی تابع
// =============================================================================
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "Generic call operation";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<F64Tensor>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

// =============================================================================
// ۶. FuncOp: تعریف تابع
// =============================================================================
def FuncOp : Toy_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "User defined function.";
  let description = [{ Defines a function in the Toy language. }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    // FunctionOpInterface Methods
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

// =============================================================================
// ۷. PrintOp
// =============================================================================
def PrintOp : Toy_Op<"print"> {
  let summary = "Print operation";
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// =============================================================================
// ۸. ReshapeOp
// =============================================================================
def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{ `(`$input`:`type($input)`)` attr-dict `to` type(results) }];
  let hasCanonicalizer = 1;
}

// =============================================================================
// ۹. ReturnOp
// =============================================================================
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<F64Tensor>:$input);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let assemblyFormat = "($input^ `:` type($input))? attr-dict";
  let hasVerifier = 1;
}

// =============================================================================
// ۱۰. TransposeOp
// =============================================================================
def TransposeOp : Toy_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Transpose operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let assemblyFormat = [{ `(`$input`:`type($input)`)` attr-dict `to` type(results) }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif // TOY_OPS
