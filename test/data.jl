using DataFrames

# dataframes:
#
function test_read_df()
    df = dataframe_read("../data/iris/iris150.csv")
    return nrow(df) == 150
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
