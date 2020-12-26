function train_tb(mdl, dtrn, dvld; opti=adam, epochs=1,
                  tb_dir=".", tb_name="tensorboard_log",
                  log_freq_trn=1, log_freq_vld=1)

    nth_mb_trn = length(dtrn) ÷ log_freq_trn
    nth_mb_vld = length(dvld) ÷ log_freq_vld

    for (imb, x)

end
