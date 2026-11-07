#include "toy/MLIRGen.h"
#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace mlir::toy;
using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// پیاده‌سازی تولید کدهای MLIR از روی درخت نحو (AST)
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// تابع اصلی برای تبدیل کل یک ماژول AST به ModuleOp در MLIR
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

  /// تبدیل موقعیت در سورس‌کد AST به Location در MLIR
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// تعریف متغیر جدید در جدول نمادها (Scope جاری)
  llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// ساخت پروتوتایپ تابع در MLIR
  mlir::toy::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, {});
    return mlir::toy::FuncOp::create(builder, location, proto.getName(),
                                     funcType);
  }

  /// ساخت تعریف و بدنه کامل تابع
  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    builder.setInsertionPointToStart(&entryBlock);

    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      ReturnOp::create(builder, loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    return function;
  }

  /// تبدیل عملیات‌های باینری (+ و *)
  mlir::Value mlirGen(BinaryExprAST &binop) {
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    switch (binop.getOp()) {
    case '+':
      return AddOp::create(builder, location, lhs, rhs);
    case '*':
      return MulOp::create(builder, location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// ارجاع به یک متغیر
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// دستور return
  llvm::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    mlir::Value expr = nullptr;
    if (auto exprOpt = ret.getExpr()) {
      if (*exprOpt) {
        if (!(expr = mlirGen(**exprOpt)))
          return mlir::failure();
      }
    }

    ReturnOp::create(builder, location,
                     expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// مقادیر تانسور ثابت چندبعدی (Literal)
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    std::vector<double> data;
    data.reserve(llvm::product_of(lit.getDims()));
    collectData(lit, data);

    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    return ConstantOp::create(builder, loc(lit.loc()), type, dataAttribute);
  }

  /// تابع بازگشتی برای مسطح‌سازی آرایه چندبعدی به بردار یک‌بعدی double
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// فراخوانی توابع (شامل transpose داخلی و generic_call)
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return TransposeOp::create(builder, location, operands[0]);
    }

    return GenericCallOp::create(builder, location, callee, operands);
  }

  /// چاپ در خروجی (print)
  llvm::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    PrintOp::create(builder, loc(call.loc()), arg);
    return mlir::success();
  }

  /// مقدار عدد ثابت اسکالر
  mlir::Value mlirGen(NumberExprAST &num) {
    return ConstantOp::create(builder, loc(num.loc()), num.getValue());
  }

  /// توزیع نوع عبارت با توجه به نوع گره AST
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Var_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Var_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Var_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Var_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Var_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// تعریف متغیر همراه با مقدار اولیه و احتمالا Reshape
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    if (!vardecl.getType().shape.empty()) {
      value = ReshapeOp::create(builder, loc(vardecl.loc()),
                                getType(vardecl.getType()), value);
    }

    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// پردازش لیست عبارات یک بلوک
  llvm::LogicalResult mlirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::failure();
        continue;
      }

      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// تبدیل شکل (Shape) به نوع Tensor در MLIR
  mlir::Type getType(ArrayRef<int64_t> shape) {
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  mlir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

/// API عمومی برای تولید کد MLIR از AST
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy
