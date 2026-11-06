//
// Created by Ali.amir on 11/6/2026.
//

#ifndef MY_TOY_PROJECT_LEXER_H
#define MY_TOY_PROJECT_LEXER_H

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <cstring>

namespace toy {
    struct Location {
        std::shared_ptr<std::string> file;
        int line;
        int col;
    };

    enum Token : int {
        tok_semicolon = ';',
        tok_parenthese_open = '(',
        tok_parenthese_close = ')',
        tok_bracket_open = '{',
        tok_bracket_close = '}',
        tok_sbracket_open = '[',
        tok_sbracket_close = ']',
        tok_eof = -1,
        tok_return = -2,
        tok_var = -3,
        tok_def = -4,
        tok_identifier = -5,
        tok_number = -6
    };

    class Lexer {

        private:
            Location          lastLocation;
            Token             curTok = tok_eof;
            std::string       identifierStr;
            double            numVal = 0;
            int               lastChar = ' ';
            int               curLineNum = 0;
            int               curCol = 0;
            llvm::StringRef   curLineBuffer = "\n";


        public:
            explicit Lexer(std::string filename) {
                lastLocation.file = std::make_shared<std::string>(std::move(filename));
                lastLocation.line = 0;
                lastLocation.col  = 0;
            }

            virtual ~Lexer() = default;

            Token getCurToken() { return curTok; }

            Token getNextToken() { return curTok = getTok(); }

            void consume(Token tok) {
                    assert(tok == curTok && "consume Token mismatch expectation");
                    getNextToken();
            }

            llvm::StringRef getId() {
                    assert(curTok == tok_identifier);
                    return identifierStr;
            }

            double getValue() {
                    assert(curTok == tok_number);
                    return numVal;
            }

            Location getLastLocation() { return lastLocation; }
            int      getLine()         { return curLineNum; }
            int      getCol()          { return curCol; }


    protected:
        virtual llvm::StringRef readNextLine() = 0;

        int getNextChar() {
            if (curLineBuffer.empty())
                return EOF;

            int c = curLineBuffer.front();
            curLineBuffer = curLineBuffer.drop_front();
            curCol++;

            if (c == '\n') {
                curLineNum++;
                curCol = 0;
                curLineBuffer = readNextLine();
            }
            return c;
        }

        Token getTok() {
            while (isspace(lastChar))
                lastChar = getNextChar();

            lastLocation.line = curLineNum;
            lastLocation.col  = curCol;

            if (isalpha(lastChar)) {
                identifierStr = static_cast<char>(lastChar);
                while (isalnum(lastChar = getNextChar()) || lastChar == '_')
                    identifierStr += static_cast<char>(lastChar);

                if (identifierStr == "def")    return tok_def;
                if (identifierStr == "return") return tok_return;
                if (identifierStr == "var")    return tok_var;
                return tok_identifier;
            }
            if (isdigit(lastChar) || lastChar == '.') {
                std::string numStr;
                do {
                    numStr += static_cast<char>(lastChar);
                    lastChar = getNextChar();
                } while (isdigit(lastChar) || lastChar == '.');

                numVal = strtod(numStr.c_str(), nullptr);
                return tok_number;
            }
            if (lastChar == '#') {
                do {
                    lastChar = getNextChar();
                } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

                if (lastChar != EOF)
                    return getTok();
            }

            if (lastChar == EOF)
                return tok_eof;

            int thisChar = lastChar;
            lastChar = getNextChar();
            return static_cast<Token>(thisChar);
        }
    };

    class LexerBuffer final : public Lexer {
        public:
            LexerBuffer(const char *begin, const char *end, std::string filename)
                : Lexer(std::move(filename)), curPtr(begin), endPtr(end) {}

        protected:
            llvm::StringRef readNextLine() override {
                if (curPtr == endPtr)
                    return llvm::StringRef();

                const char *newLine = static_cast<const char *>(
                    std::memchr(curPtr, '\n', static_cast<size_t>(endPtr - curPtr)));


                const char *newEnd = newLine ? newLine + 1 : endPtr;

                llvm::StringRef line(curPtr, static_cast<size_t>(newEnd - curPtr));
                curPtr = newEnd;
                return line;
            }

        private:
            const char *curPtr;
            const char *endPtr;
        };

}

#endif //MY_TOY_PROJECT_LEXER_H
