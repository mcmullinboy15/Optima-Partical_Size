"""
Find Used_Pred Plan and Ideas

predictors = [
    pred_01.pth,
    pred_02.pth,
    pred_03.pth,
    ...
]

preds = [
    Pred_01,
    Pred_02,
    Pred_03,
    ...
]

Option 1:
    when Real_Man_Passing changes
        UsePredictor = getLowestLossPredictor(preds)
Option 2:


"""