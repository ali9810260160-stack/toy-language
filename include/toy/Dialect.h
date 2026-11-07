//
// Created by Ali.amir on 11/7/2026.
//

#ifndef MY_TOY_PROJECT_DIALECT_H
#define MY_TOY_PROJECT_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif //MY_TOY_PROJECT_DIALECT_H
