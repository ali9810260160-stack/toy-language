#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A tutorial Toy dialect";
  let description = [{
    The Toy dialect is a simple dialect designed to demonstrate the
    capabilities of MLIR. It defines a simple tensor-based language.
  }];
  let cppNamespace = "::mlir::toy";
}


class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;


def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "Constant operation";
  let description = [{ Defines a constant tensor. }];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "double":$value), [{
      build($_builder, $_state, RankedTensorType::get({}, $_builder.getF64Type()),
            $_builder.getDenseF64ArrayAttr(value));
    }]>,
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def AddOp : Toy_Op<"add", [Pure]> {
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

def MulOp : Toy_Op<"mul", [Pure]> {
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


def GenericCallOp : Toy_Op<"generic_call"> {
  let summary = "Generic call operation";
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments), [{
      build($_builder, $_state, UnrankedTensorType::get($_builder.getF64Type()),
            $_builder.getStringAttr(callee), arguments);
    }]>
  ];

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}


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
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type), [{
      build($_builder, $_state, $_builder.getStringAttr(name), TypeAttr::get(type));
    }]>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the current operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}


def PrintOp : Toy_Op<"print"> {
  let summary = "Print operation";
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}


def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{ `(`$input`:`type($input)`)` attr-dict `to` type(results) }];
}


def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "Return operation";
  let arguments = (ins Variadic<F64Tensor>:$input);
  let assemblyFormat = "($input^ `:` type($input))? attr-dict";
  let hasVerifier = 1;
}


def TransposeOp : Toy_Op<"transpose", [Pure]> {
  let summary = "Transpose operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
  let assemblyFormat = [{ `(`$input`:`type($input)`)` attr-dict `to` type(results) }];
  let hasVerifier = 1;
}

#endif // TOY_OPS