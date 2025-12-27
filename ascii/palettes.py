##
## ASCII CHARACTER PALETTES 
##
NUMBERS = ["0","1","2","3","4","5","6","7","8","9"]
OPERATORS = ['+','-','±','=','≠','≈','*','/','•','÷','~','^','%','|','&','<','>','≤','≥','√','∫']
PUNCTUATION = [' ','.',',',"'",'"','-','\\','/','_',':',';','?','¿','!','¡','~','`','´','…','·','¯']
SYMBOLS = ['@','#','$','%','&']
CURRENCIES = ['$','¥','£','¢']
MISC = ['·','˙','ˇ','ˆ','¨','¯','˘','‹','›','¸','˛','«','†','˜','°','©','®','™']
BRACKETS = ['[',']','{','}','(',')','<','>']
ENGLISH_ALPHABET = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z']
GREEK_ALPHABET = ['Α', 'α', 'Β', 'β', 'Γ', 'γ', 'Δ', 'δ', 'Ε', 'ε', 'Ζ', 'ζ', 'Η', 'η', 'Θ', 'θ', 'Ι', 'ι', 'Κ', 'κ', 'Λ', 'λ', 'Μ', 'μ', 'Ν', 'ν', 'Ξ', 'ξ', 'Ο', 'ο', 'Π', 'π', 'Ρ', 'ρ', 'Σ', 'σ', 'ς', 'Τ', 'τ', 'Υ', 'υ', 'Φ', 'φ', 'Χ', 'χ', 'Ψ', 'ψ', 'Ω', 'ω']
# BLOCKS = ['█','▓','▒','▌','▐','▖','▗','▘','▙','▛','▜','▟']

DEFAULT = list(set(
    ENGLISH_ALPHABET + PUNCTUATION + NUMBERS + OPERATORS + SYMBOLS + BRACKETS
))
ALL = list(set(
    NUMBERS + OPERATORS + PUNCTUATION + SYMBOLS + CURRENCIES + MISC + BRACKETS + ENGLISH_ALPHABET + GREEK_ALPHABET 
))

PALETTES = {
    "default": DEFAULT,
    "all": ALL,
    "numbers": NUMBERS,
    "operators": OPERATORS,
    "math": NUMBERS+OPERATORS,
    "punctuation": PUNCTUATION,
    "symbols": SYMBOLS,
    "currencies": CURRENCIES,
    "misc": MISC,
    "brackets": BRACKETS,
    "english": ENGLISH_ALPHABET,
    "greek": GREEK_ALPHABET,
    "book": ENGLISH_ALPHABET+PUNCTUATION,
}

