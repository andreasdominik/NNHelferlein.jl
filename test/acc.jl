# funs for test of acc-funs:
#

function test_peak_finder()

    t = [0, 0,   0,   0, 0,   1,   0,   0,   0, 0, 0, 1]
    y = [0, 0.1, 0.1, 0, 0.8, 0,   0,   0, 0.9, 0, 0, 0]

    #t = KnetArray(t)
    #p = KnetArray(p)
    f1 = peak_finder_acc(y, t, verbose=3)  # != 2/3
    gm = peak_finder_acc(y, t, verbose=1, ret=:g_mean)  # ≈ 0.7071
    iou = peak_finder_acc(y, t, verbose=1, ret=:iou)  # 
    return f1 ≈ 2/3 && gm ≈ 0.7071067811865 && iou ≈ 0.5
end



function test_peak_finder_acc()

    t = [0, 0,   0,   0, 0,   1,   0,   0,   0, 0, 0, 1]
    y = [0, 0.1, 0.1, 0, 0.8, 0,   0,   0, 0.9, 0, 0, 0]
    d = [(y,t), (y,t)]

    mdl(x) = x
    f1 = peak_finder_acc(mdl, data=d)
    return f1 #≈ 2/3 
end


