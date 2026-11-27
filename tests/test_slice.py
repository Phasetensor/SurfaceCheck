from surfacecheck import check_slice


def test_slice():
    k = [90, 100, 110]
    assert check_slice(k, [12, 8, 4], [12, 8, 4])["status"] == "mid_feasible"
    assert check_slice(k, [11, 7, 3], [13, 11, 5])["status"] == "spreads_feasible"
    assert check_slice(k, [12, 9, 4], [12, 9, 4])["status"] == "spreads_infeasible"
