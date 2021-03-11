# test NLP utils:
#

function test_tokenizer()

    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    return tok("Julia") == 4 && tok(4) == "Julia"
end

function test_seq_mb()

    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])

    t = [tok("I love Julia", split_words=true),
         tok("Peter loves Python", split_words=true),
         tok("Peter loves Julia and I", split_words=true)]

    mb = seq_minibatch(t, 2, seq_len=4)
    return size(first(mb)) == (4,2)
end
