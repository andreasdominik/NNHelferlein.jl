using DataFrames

# dataframes:
#
function test_read_df()
    df = dataframe_read("../data/iris/iris150.csv")
    return nrow(df) == 150
end

function test_df_loader()

        trn = DataFrame(x1=randn(16), x2=randn(16),
                        x3=randn(16), x4=randn(16),
                        x5=randn(16), x6=randn(16),
                        x7=randn(16), x8=randn(16),
                        y=["blue", "red", "green", "green",
                           "blue", "red", "green", "green",
                           "blue", "red", "green", "green",
                           "blue", "red", "green", "green"])

        mb = dataframe_minibatches(trn, size=4, teaching="y", ignore="x1")
        return first(mb)[2] == [0x01  0x03  0x02  0x02]
end



function test_df_split()
    df = dataframe_read("../data/iris/iris150.csv")
    t,v = dataframe_split(df, fr=0.5, shuffle=true,
          teaching="species", balanced=true)
    return nrow(v) == 75
end

function test_df_class_ids()
    df = dataframe_read("../data/iris/iris150.csv")
    c = mk_class_ids(df.species)
    return c[2] == ["setosa", "versicolor", "virginica"]
end

function test_df_minibatch()
    df = dataframe_read("../data/iris/iris150.csv")
    mb = dataframe_minibatches(df, size=10, teaching="species")
    return size(first(mb)[1]) == (4,10)
end


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


# test tatoeba on tiny dataset Galician:
#
function test_tatoeba()
    en,glg = get_tatoeba_corpus("glg")
    return en isa AbstractArray
end

function test_seq2seq_mb()
     en,glg = get_tatoeba_corpus("glg")
     ven = WordTokenizer(en)
     vglg = WordTokenizer(glg)
     mb = seq2seq_minibatch(ven.(en, split_words=true),
            vglg.(glg, split_words=true),
            32, seq_len=20,
            pad_x=ven("<pad>"), pad_y=vglg("<pad>"))

    mb1 = first(mb)[1]
    return size(mb1) == (20,32)
end
