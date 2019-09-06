import pandas as pd

from . import compare_captions


def test_ranks():
    compare_captions.initialize()  # 15.98s

    contest = 530
    captions = [
        "Your overhead is going to kill you.",  # 1
        "The latest polls show you hanging on by a thread.",  # 50
        "First off, we need to renew your life insurance policy.",  # 100
        "And lastly, your approval ratings are hanging by a thread. No pun intended.",  # 332
        "The queen fears you are overcompensating",  # 516
        "...and finally, Your Majesty, please don't kill the messenger.",  # 792
        "The instructions from Damocles is quite clear, your highness. The sword IS supposed to hang over your head.",  # 1092
        "May God grant you great bris.",  # 1357
        "Sire, it's from your interior designer. He says he's tired of hearing \"Your check is in the mail, Damocles.\"",  # 3339
        "The Council has decreed it essential that no ruler should feel at ease in his position. And since it's too late to re-design the throne itself....",  # 2764
    ]
    ranks = compare_captions.rank_captions(captions, contest)
    assert isinstance(ranks, pd.Series)
    assert ranks.index[0] == captions[0]
    assert ranks.index[1] == captions[1]
    assert ranks.index[3] == captions[2]  # inaccuracy
    assert ranks.index[-1] == captions[-2]
    assert ranks.index[-2] == captions[-4]

    assert 0 <= ranks.min() <= ranks.max() <= 1
