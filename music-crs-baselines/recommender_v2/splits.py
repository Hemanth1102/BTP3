"""Session indices in the TalkPlayData Challenge train and test splits."""

TRAIN_SPLIT = "train"
DEV_SPLIT = "test"

TRAIN = range(15_199)
DEV_TRAIN = (range(200), range(600, 1_000))
FINAL_HOLDOUT = range(200, 400)
DIAGNOSTIC = range(400, 600)

TURNS = range(1, 9)
