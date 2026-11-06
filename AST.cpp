//
// Created by Ali.amir on 11/6/2026.
//

#include "toy/AST.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>

using namespace toy;

namespace {
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
  void indent() {
    for (int i = 0; i < curIndent; ++i)
      llvm::errs() << "  ";
  }

  void dump(ExprAST *expr);
  void dump(const VarType &type);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(VarDeclExprAST *varDecl);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);

  int curIndent = 0;
};
} // namespace


void ASTDumper::dump(ExprAST *expr) {
  llvm::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

void ASTDumper::dump(const VarType &type) {
  llvm::errs() << "<";
  llvm::interleaveComma(type.shape, llvm::errs());
  llvm::errs() << ">";
}

void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  llvm::errs() << " (";
  auto loc = varDecl->loc();
  llvm::errs() << *loc.file << ":" << loc.line << ":" << loc.col << ")\n";

  if (auto *initVal = varDecl->getInitVal()) {
    Indent level_(curIndent);
    dump(initVal);
  }
}

void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  llvm::errs() << "Block {\n";
  {
    Indent level_(curIndent);
    for (auto &expr : *exprList)
      dump(expr.get());
  }
  indent();
  llvm::errs() << "}\n";
}

void ASTDumper::dump(NumberExprAST *num) {
  indent();
  llvm::errs() << "Num: " << num->getValue() << "\n";
}

void ASTDumper::dump(LiteralExprAST *node) {
  indent();
  llvm::errs() << "Literal: ";

  std::function<void(ExprAST*)> printHelper = [&](ExprAST* litOrNum) {
    if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
      llvm::errs() << num->getValue();
      return;
    }
    auto *literal = llvm::cast<LiteralExprAST>(litOrNum);
    llvm::errs() << "<";
    llvm::interleaveComma(literal->getDims(), llvm::errs());
    llvm::errs() << ">[";
    llvm::interleaveComma(literal->getValues(), llvm::errs(),
                          [&](auto &elt) { printHelper(elt.get()); });
    llvm::errs() << "]";
  };

  printHelper(node);
  llvm::errs() << "\n";
}

void ASTDumper::dump(VariableExprAST *node) {
  indent();
  llvm::errs() << "Var: " << node->getName() << "\n";
}

void ASTDumper::dump(ReturnExprAST *node) {
  indent();
  llvm::errs() << "Return\n";
  if (node->getExpr().has_value()) {
    Indent level_(curIndent);
    dump(*node->getExpr());
  }
}

void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  llvm::errs() << "BinOp: " << node->getOp() << "\n";
  {
    Indent level_(curIndent);
    dump(node->getLHS());
    dump(node->getRHS());
  }
}

void ASTDumper::dump(CallExprAST *node) {
  INDENT();
  llvm::errs() << "Call '" << node->getCallee() << "':\n";
  {
    Indent level_(curIndent);
    for (auto &arg : node->getArgs())
      dump(arg.get());
  }
}

void ASTDumper::dump(PrintExprAST *node) {
  INDENT();
  llvm::errs() << "Print [\n";
  {
    Indent level_(curIndent);
    dump(node->getArg());
  }
  indent();
  llvm::errs() << "]\n";
}

void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  llvm::errs() << "Proto '" << node->getName() << "' (";
  llvm::interleaveComma(node->getArgs(), llvm::errs(),
                        [&](auto &arg) { llvm::errs() << arg->getName(); });
  llvm::errs() << ")\n";
}

void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  llvm::errs() << "Function \n";
  {
    Indent level_(curIndent);
    dump(node->getProto());
    dump(node->getBody());
  }
}

void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &f : *node)
    dump(&f);
}


namespace toy {
void dump(ModuleAST &module) { ASTDumper().dump(&module); }
} // namespace toy