#!/bin/bash
set -e

# TinyTeX PATH
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Compiling pai_airoa_04.tex with LuaLaTeX..."
lualatex -interaction=nonstopmode pai_airoa_04.tex
lualatex -interaction=nonstopmode pai_airoa_04.tex

echo "Done! Output: $SCRIPT_DIR/pai_airoa_04.pdf"

