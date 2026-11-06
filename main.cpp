#include "toy/Lexer.h"

#include <iostream>
#include <string>

static std::string tokenKindName(toy::Token t) {
    switch (t) {
        case toy::tok_eof:        return "tok_eof";
        case toy::tok_def:        return "tok_def";
        case toy::tok_return:     return "tok_return";
        case toy::tok_var:        return "tok_var";
        case toy::tok_identifier: return "tok_identifier";
        case toy::tok_number:     return "tok_number";
        default:
            if (t >= 32 && t < 127) {
                std::string s = "char '";
                s += static_cast<char>(t);
                s += "'";
                return s;
            }
            return "unknown(" + std::to_string(static_cast<int>(t)) + ")";
    }
}


int main() {
    std::string src = "def main() { var a = [1, 2]; }";

    toy::LexerBuffer lexer(src.data(), src.data() + src.size(), "test.toy");

    toy::Token tok;
    while ((tok = lexer.getNextToken()) != toy::tok_eof) {
        toy::Location loc = lexer.getLastLocation();

        std::cout << *loc.file << ":" << loc.line << ":" << loc.col
                  << ": " << tokenKindName(tok);

        if (tok == toy::tok_identifier)
            std::cout << "  (text = \"" << lexer.getId().str() << "\")";
        else if (tok == toy::tok_number)
            std::cout << "  (value = " << lexer.getValue() << ")";

        std::cout << "\n";
    }

    std::cout << "==> reached tok_eof; tokenization done.\n";
    return 0;
}