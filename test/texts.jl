# test NLP utils:
#

function test_tokenizer()

    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    return tok("Julia") == 4 && tok(4) == "Julia"
end
