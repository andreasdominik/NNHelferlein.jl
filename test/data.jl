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

        mb1 = dataframe_minibatches(trn, size=4, teaching="y", ignore="x1")

        trn.iy = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        @show mb2 = dataframe_minibatches(trn, size=4, teaching="iy", ignore=["x1", "y"])
        @show first(mb2)[2]

        trn.is = [:a,:a,:a,:a,:b,:b,:b,:b,:c,:c,:c,:c,:d,:d,:d,:d]
        mb3 = dataframe_minibatches(trn, size=4, teaching="is", ignore=["iy", "y"])

        return first(mb1)[2] == UInt8[0x01  0x03  0x02  0x02] &&
               first(mb2)[2] == UInt8[0x01 0x01 0x01 0x01]    &&
               isnothing(mb3)
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

function test_df_errors()
    df1 = dataframe_read("../data/iris/iris150.ods")
    df2 = dataframe_read("../data/iris/iris150.dat")

    return isnothing(df1) &&
           isnothing(df2)
end

# test NLP utils:
#
function test_tokenizer()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])
    l = tok(["I love Julia", "Peter loves Python", "We all marvel Geoff"],
            add_ctls=true)

    sentence = tok([8, 12, 5])
    return tok("Julia") == 8 && tok(8) == "Julia" &&
           l[3] == [1, 6, 14, 13, 11, 2] &&
           sentence == "Julia loves Peter"
end

function test_seq_mb()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])

    t = [tok("I love Julia", split_words=true),
         tok("Peter loves Python", split_words=true),
         tok("Peter loves Julia and Scala", split_words=true)]

    mb = seq_minibatch(t, 2, seq_len=4)
    return size(first(mb)) == (4,2)
end

function test_seq_mb_xy()
    tok = WordTokenizer(["I love Julia",
                         "Peter loves Python",
                         "We all marvel Geoff"])

    t = [tok("I love Julia", split_words=true),
         tok("Peter loves Python", split_words=true),
         tok("Peter loves Julia and Scala", split_words=true)]

    y = [1 1 2]
    mb = seq_minibatch(t, y, 2, seq_len=4)
    return first(mb)[2] == [1 1]
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
