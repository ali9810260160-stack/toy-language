//
// Created by Ali.amir on 11/6/2026.
//

#ifndef MY_TOY_PROJECT_PARSER_H
#define MY_TOY_PROJECT_PARSER_H

#include "toy/AST.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace toy {

  class Parser {
    Lexer &lexer;

    public:
      Parser(Lexer &lexer) : lexer(lexer) {}


      std::unique_ptr<ModuleAST> parseModule() {
        lexer.getNextToken();

        std::vector<FunctionAST> functions;
        while (lexer.getCurToken() != tok_eof) {
          auto fn = parseDefinition();
          if (!fn)
            return nullptr;
          functions.push_back(std::move(*fn));
        }
        return std::make_unique<ModuleAST>(std::move(functions));
      }

    private:
      template <typename R, typename T, typename U = const char *>
      std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
        auto curToken = lexer.getCurToken();
        llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                     << lexer.getLastLocation().col << "): expected '" << expected
                     << "' " << context << " but has Token " << curToken;
        if (isprint(static_cast<unsigned char>(curToken)))
          llvm::errs() << " '" << (char)curToken << "'";
        llvm::errs() << "\n";
        return nullptr;
      }

      std::unique_ptr<ExprAST> parseNumberExpr() {
        auto loc = lexer.getLastLocation();
        auto result = std::make_unique<NumberExprAST>(loc, lexer.getValue());
        lexer.consume(tok_number);
        return result;
      }

      std::unique_ptr<ExprAST> parseParenExpr() {
        lexer.consume(static_cast<Token>('('));
        auto v = parseExpression();
        if (!v)
          return nullptr;

        if (lexer.getCurToken() != ')')
          return parseError<ExprAST>("')'");

        lexer.consume(static_cast<Token>(')'));
        return v;
      }

      std::unique_ptr<ExprAST> parseIdentifierExpr() {
        std::string name = lexer.getId().str();
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);

        if (lexer.getCurToken() != '(')
          return std::make_unique<VariableExprAST>(loc, name);

        lexer.consume(static_cast<Token>('('));
        std::vector<std::unique_ptr<ExprAST>> args;
        if (lexer.getCurToken() != ')') {
          while (true) {
            if (auto arg = parseExpression())
              args.push_back(std::move(arg));
            else
              return nullptr;

            if (lexer.getCurToken() == ')')
              break;

            if (lexer.getCurToken() != ',')
              return parseError<ExprAST>("',' or ')'");

            lexer.consume(static_cast<Token>(','));
          }
        }
        lexer.consume(static_cast<Token>(')'));

        if (name == "print") {
          if (args.size() != 1)
            return parseError<ExprAST>("1 argument for print");
          return std::make_unique<PrintExprAST>(loc, std::move(args[0]));
        }

        return std::make_unique<CallExprAST>(loc, name, std::move(args));
      }

      std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
        auto loc = lexer.getLastLocation();
        lexer.consume(static_cast<Token>('['));

        std::vector<std::unique_ptr<ExprAST>> values;
        std::vector<int64_t> dims;
        bool first = true;

        while (lexer.getCurToken() != ']') {
          if (!first) {
            if (lexer.getCurToken() != ',')
              return parseError<ExprAST>("',' or ']'");
            lexer.consume(static_cast<Token>(','));
          }
          first = false;

          std::unique_ptr<ExprAST> value;
          if (lexer.getCurToken() == '[') {
            value = parseTensorLiteralExpr();
          } else {
            value = parseNumberExpr();
          }
          if (!value)
            return nullptr;

          if (values.empty()) {
            if (auto *lit = llvm::dyn_cast<LiteralExprAST>(value.get())) {
              dims = lit->getDims().vec();
            }
          } else {
            if (auto *lit = llvm::dyn_cast<LiteralExprAST>(value.get())) {
              if (lit->getDims() != llvm::ArrayRef<int64_t>(dims))
                return parseError<ExprAST>("uniform tensor");
            } else {
              if (!dims.empty())
                return parseError<ExprAST>("uniform tensor");
            }
          }
          values.push_back(std::move(value));
        }
        lexer.consume(static_cast<Token>(']'));

        dims.insert(dims.begin(), values.size());
        return std::make_unique<LiteralExprAST>(loc, std::move(values),
                                                std::move(dims));
      }

      std::unique_ptr<ExprAST> parsePrimary() {
        switch (static_cast<int>(lexer.getCurToken())) {
        default:
          return parseError<ExprAST>("an expression");
        case tok_identifier:
          return parseIdentifierExpr();
        case tok_number:
          return parseNumberExpr();
        case '(':
          return parseParenExpr();
        case '[':
          return parseTensorLiteralExpr();
        }
      }

      int getTokPrecedence() {
        if (!isascii(lexer.getCurToken()))
          return -1;

        switch (static_cast<int>(lexer.getCurToken())) {
        case '+':
        case '-':
          return 20;
        case '*':
          return 40;
        default:
          return -1;
        }
      }

      std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                             std::unique_ptr<ExprAST> lhs) {
        while (true) {
          int tokPrec = getTokPrecedence();

          if (tokPrec < exprPrec)
            return lhs;

          int binOp = lexer.getCurToken();
          auto loc = lexer.getLastLocation();
          lexer.consume(static_cast<Token>(binOp));

          auto rhs = parsePrimary();
          if (!rhs)
            return nullptr;

          int nextPrec = getTokPrecedence();
          if (tokPrec < nextPrec) {
            rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
            if (!rhs)
              return nullptr;
          }

          lhs = std::make_unique<BinaryExprAST>(loc, binOp, std::move(lhs),
                                                std::move(rhs));
        }
      }

      std::unique_ptr<ExprAST> parseExpression() {
        auto lhs = parsePrimary();
        if (!lhs)
          return nullptr;
        return parseBinOpRHS(0, std::move(lhs));
      }

      std::unique_ptr<VarType> parseType() {
        if (lexer.getCurToken() != '<')
          return parseError<VarType>("'<'");
        lexer.consume(static_cast<Token>('<'));

        auto type = std::make_unique<VarType>();
        while (lexer.getCurToken() != '>') {
          if (!type->shape.empty()) {
            if (lexer.getCurToken() != ',')
              return parseError<VarType>("',' or '>'");
            lexer.consume(static_cast<Token>(','));
          }

          if (lexer.getCurToken() != tok_number)
            return parseError<VarType>("number");

          type->shape.push_back(lexer.getValue());
          lexer.consume(tok_number);
        }
        lexer.consume(static_cast<Token>('>'));
        return type;
      }

      std::unique_ptr<VarDeclExprAST> parseDeclaration() {
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_var);

        if (lexer.getCurToken() != tok_identifier)
          return parseError<VarDeclExprAST>("identifier");

        std::string name = lexer.getId().str();
        lexer.consume(tok_identifier);

        VarType type;
        if (lexer.getCurToken() == '<') {
          auto parsedType = parseType();
          if (!parsedType)
            return nullptr;
          type = std::move(*parsedType);
        }

        if (lexer.getCurToken() != '=')
          return parseError<VarDeclExprAST>("'='");
        lexer.consume(static_cast<Token>('='));

        auto initVal = parseExpression();
        if (!initVal)
          return nullptr;

        return std::make_unique<VarDeclExprAST>(loc, name, std::move(type),
                                                std::move(initVal));
      }

      std::unique_ptr<ReturnExprAST> parseReturn() {
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_return);

        std::optional<std::unique_ptr<ExprAST>> expr;
        if (lexer.getCurToken() != ';') {
          auto e = parseExpression();
          if (!e)
            return nullptr;
          expr = std::move(e);
        }
        return std::make_unique<ReturnExprAST>(loc, std::move(expr));
      }

      std::unique_ptr<ExprASTList> parseBlock() {
        if (lexer.getCurToken() != '{')
          return parseError<ExprASTList>("'{'");
        lexer.consume(static_cast<Token>('{'));

        auto body = std::make_unique<ExprASTList>();
        while (lexer.getCurToken() != '}') {
          std::unique_ptr<ExprAST> expr;
          switch (lexer.getCurToken()) {
          case tok_var:
            expr = parseDeclaration();
            break;
          case tok_return:
            expr = parseReturn();
            break;
          default:
            expr = parseExpression();
            break;
          }
          if (!expr)
            return nullptr;

          body->push_back(std::move(expr));

          if (lexer.getCurToken() != ';')
            return parseError<ExprASTList>("';'");
          lexer.consume(static_cast<Token>(';'));
        }
        lexer.consume(static_cast<Token>('}'));
        return body;
      }

      std::unique_ptr<PrototypeAST> parsePrototype() {
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_def);

        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>("function name");

        std::string name = lexer.getId().str();
        lexer.consume(tok_identifier);

        if (lexer.getCurToken() != '(')
          return parseError<PrototypeAST>("'('");
        lexer.consume(static_cast<Token>('('));

        std::vector<std::unique_ptr<VariableExprAST>> args;
        while (lexer.getCurToken() != ')') {
          if (!args.empty()) {
            if (lexer.getCurToken() != ',')
              return parseError<PrototypeAST>("',' or ')'");
            lexer.consume(static_cast<Token>(','));
          }
          if (lexer.getCurToken() != tok_identifier)
            return parseError<PrototypeAST>("identifier");

          auto argLoc = lexer.getLastLocation();
          args.push_back(
              std::make_unique<VariableExprAST>(argLoc, lexer.getId()));
          lexer.consume(tok_identifier);
        }
        lexer.consume(static_cast<Token>(')'));

        return std::make_unique<PrototypeAST>(loc, name, std::move(args));
      }

      std::unique_ptr<FunctionAST> parseDefinition() {
        auto proto = parsePrototype();
        if (!proto)
          return nullptr;

        if (auto body = parseBlock())
          return std::make_unique<FunctionAST>(std::move(proto), std::move(body));

        return nullptr;
      }
  };

} // namespace toy

#endif //MY_TOY_PROJECT_PARSER_H
