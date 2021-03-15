import pprint
import sys

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from Test import Test
from model_and_defs import Model1, Model2, Model3, calculate_Accuracy, plot # , calculate_Accuracy, plot, getFileName


# pyodbc
# PDBC
# OPC

"""
pyinstaller --onefile Use_model.py

pyinstaller --hiddenimport pywt._extensions._cwt myScript.py

--exclude-module:
 matplotlib

"""
data = f'data\Compiled_Data.csv'

tests = [
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_100.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_800.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_1600.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_3200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_6400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_10_12800.pth', data),

    Test('Outputs/Epochs/Andrews_Model1_128_256_50_100.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_800.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_1600.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_3200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_6400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_50_12800.pth', data),

    Test('Outputs/Epochs/Andrews_Model1_128_256_100_100.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_800.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_1600.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_3200.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_6400.pth', data),
    Test('Outputs/Epochs/Andrews_Model1_128_256_100_12800.pth', data),
]
tests_ = {}

for test in tests:
    test._test()
    # test.plot()
    tests_.update({f"{test.loaded_dict['epochs']}_{test.loaded_dict['BATCH_SIZE']}" : { 'R2':test.loaded_dict['R2'], 'MSE':test.loaded_dict['MSE']}})

pprint.PrettyPrinter(indent=4).pprint(tests_)
sys.exit(-23)



D_in = 16
D_out = 1

# H1 = 64
# H2 = 128
# H3 = 256
# Model = Model2

H1 = 128
H2 = None
H3 = None
Model = Model1

#

ROUND_TO = 3
SAVE=False


model = Model(D_in, H1, D_out, H2=H2, H3=H3)
model.load_state_dict(torch.load(model_path)) #['model_state_dict'])
model.eval()
# print(model)

file = "data/Collected_1-1_6-10-2020 - CLEANED_DATA.csv"
# file = "data/AAPL-5-minute-2020-01-01-2020-02-01.csv"

df = pd.read_csv(file)#, index_col=0)
df = df[:200]
# print(df)
data = np.array(df).astype(dtype="float64")



x_test = data[:, 1:]
y_test = data[:, 0]
y_test = y_test.reshape(-1, 1)


with torch.no_grad():

    # if torch.cuda.is_available():
    # predicted = model(torch.from_numpy(x_test).cuda().float()).cpu() # .data.numpy()
    # else:

    preds = []

    for test in x_test:
        print(test)
        print(type(test))

        test1 = test.reshape(-1, 1)
        print(test1)
        print(type(test1))

        s = MinMaxScaler(feature_range=(0, 1))
        test2 = s.fit_transform(test1)

        test3 = test2.reshape(1, -1)
        print(test3)
        print(type(test3))
        predicted = model(torch.from_numpy(test3).float())
        print(predicted)
        print(type(predicted))
        print(predicted.shape)

        pred1 = predicted.numpy()
        print(pred1)
        print(type(pred1))
        print(pred1.shape)

        pred2 = s.inverse_transform(pred1)
        print(pred2)
        preds.append(pred2.item())
        print()
        print()

    print(preds)

predicted = torch.from_numpy(np.asarray(preds))

# This is where I Calculate Accuracy
calculate_Accuracy(predicted, y_test, ROUND_TO, SAVE)


# plotting the graph
plot(predicted, y_test, "True data", 'go', "Predictions", 'x', SAVE)


for i in range(len(x_test)):
    _5_min_guess = model(torch.from_numpy(x_test[i]).float()).data.numpy(), #, .item()
    y = y_test[i]
    # print(y, _5_min_guess, "\t", (y.item() - _5_min_guess.item()))

_data = [[ 162.07161823, 4975.19557879, 631.74528143, 527.04292352, 599.82067682,
           12.43808911, 58.07202386, 1512.49702026, 314.92916092, 1.,
           338.9874533, 618.18067335, 5.57599382, 2975.35850815, 70., 8. ],
        [ 161.66513192, 4975.19495146, 631.99421063, 527.04255589, 599.82694682,
          12.43709558, 58.07007837, 1512.50532244, 314.9420118, 1.,
          338.67214583, 618.18274885, 5.57606108, 2975.39239103, 70., 8. ],
        [ 161.47963145, 4975.1944969, 631.99705234, 527.04218847, 599.82714978,
          12.43071289, 58.06977684, 1512.51374032, 314.94205834, 1.,
          338.6629651, 618.18849368, 5.57612723, 2975.58037931, 70., 8. ],
        [ 161.38310402, 4975.1939655, 631.99989232, 527.05488478, 599.82735438,
          12.42966203, 58.06180724, 1512.52227388, 314.94210159, 1.,
          338.65356087, 618.32026384, 5.57619226, 2975.61778579, 70., 8. ],
        [ 161.46702452, 4975.1933509, 632.00272943, 527.05458075, 599.82756059,
          12.42858362, 58.06978978, 1508.34809391, 314.94214135, 1.,
          338.64328777, 618.18456827, 5.57625617, 2975.65540734, 70., 8. ],
        [ 161.56086377, 4975.19289851, 632.00556254, 527.0542752, 599.82776838,
          12.42747707, 58.07217915, 1508.28182802, 314.94217741, 1.,
          338.63273869, 618.17770931, 5.57601878, 2975.6932452, 70., 8. ],
        [ 161.46612835, 4975.43498848, 632.00839046, 527.05396807, 599.82797768,
          12.43122331, 58.06372284, 1508.21396312, 314.94220955, 1.,
          338.62190747, 617.8979681, 5.5760898, 2975.73130062, 70., 8. ],
        [ 161.56422401, 4975.4289077, 632.22458291, 527.05365929, 599.82824724,
          12.43005891, 58.07025407, 1508.1444615, 314.93122319, 1.,
          338.62770258, 618.02003374, 5.57615985, 2975.76957493, 70., 8. ],
        [ 161.56349839, 4975.42275659, 632.22751024, 527.05334881, 599.82852412,
          12.42886611, 58.07359121, 1508.07328474, 314.9312468, 1.,
          338.61656544, 617.86917372, 5.57622891, 2975.94434002, 70., 8. ],
        [ 161.5642245, 4975.41581078, 632.23043246, 527.05149967, 599.82880851,
          12.42773343, 58.074922, 1508.00039367, 314.93127125, 1.,
          338.60513152, 618.13581629, 5.57629697, 2975.98565084, 70., 8. ],
        [ 161.862427, 4975.41049341, 631.75123103, 527.01879261, 599.83229056
        , 12.41196085, 58.04466502, 1505.51996917, 314.93739899, 1.
        , 338.3584986, 617.54670159, 5.58379787, 2980.36568664, 70.
        , 7.94651424],
        [ 161.86505129, 4975.74488456, 631.75415581, 527.01859932, 599.83231367
        , 12.41161608, 58.04830904, 1505.83655824, 314.93748142, 1.
        , 338.35549511, 617.66645903, 5.58384363, 2980.42001617, 70.
        , 7.94642245],
        [ 161.87264168, 4975.75812703, 631.75712919, 527.01840906, 599.83233643
        , 12.41125217, 58.04443874, 1505.80929223, 314.93756769, 1.
        , 338.35231831, 617.64642048, 5.58593396, 2980.47469947, 70.
        , 7.94634204],
        [ 161.87241226, 4975.75549962, 631.76015181, 527.01822196, 599.81750222
        , 12.41087115, 58.04388317, 1505.78104007, 314.93765793, 1.
        , 338.34896021, 617.49742981, 5.58604034, 2980.52974257, 70.
        , 7.94627338],
        [ 161.78105847, 4975.75268143, 631.96814239, 527.01803814, 599.81728055
        , 12.41045723, 58.03468043, 1505.75175704, 314.9267379, 1.
        , 338.32086593, 617.74698487, 5.58614658, 2980.5851517, 70.
        , 7.9462161, ],
        [ 161.77933662, 4975.76625414, 631.97103463, 527.01785771, 599.81715933
        , 12.41003593, 58.03722118, 1505.72139699, 314.92686581, 1.
        , 338.31676048, 617.6031372, 5.58625265, 2980.66078467, 70.
        , 7.946173, ],
        [ 161.77639125, 4975.76507026, 631.97395863, 527.01896156, 599.81703521
        , 12.41158744, 58.04195415, 1505.68991234, 314.92699851, 1.
        , 338.31270058, 617.44781461, 5.58635851, 2980.71501904, 70.
        , 7.94614211],
        [ 161.86140978,4975.71950292, 631.97691526, 527.01871695, 599.8169081
        , 12.41120356, 58.04169737, 1505.6157741, 314.92713615, 1.
        , 338.3086898, 617.29079002, 5.58646413, 2980.76957055, 70.
        , 7.94612468],
        [ 161.94480451, 4975.71790523, 631.97990543, 527.01847106, 599.8167779
        , 12.4108306, 58.03653091, 1505.57254075, 314.92727884, 1.
        , 338.30473182, 617.12853372, 5.58861461, 2980.824444, 70.
        , 7.94612116],
        [ 161.93758401, 4975.71618718, 631.98293007, 527.01822381, 599.81664452
        , 12.41046723, 58.0484526, 1505.52973484, 314.92742674, 1.
        , 338.3008285, 617.37814201, 5.58876611, 2980.87964427, 70.
        , 7.94613199],
        [ 161.83569571, 4975.71541948, 632.1435209, 527.01797513, 599.81656243
        , 12.41011079, 58.03707791, 1505.48738417, 314.92757998, 1.
        , 338.28885203, 617.49943395, 5.58891911, 2980.93517638, 70.
        , 7.94615766],
        [ 161.83766281, 4975.71448671, 632.14550469, 527.01772492, 599.81647874
        , 12.40975008, 58.03370011, 1505.44551738, 314.92755815, 1.
        , 338.28433321, 617.62191177, 5.58907371, 2980.9593912, 70.
        , 7.94619865],
        [ 161.83964861, 4975.72833518, 632.14751339, 527.02080304, 599.81639509
        , 12.41024929, 58.03707621, 1505.40416395, 314.92753948, 1.
        , 338.27976925, 617.87583963, 5.58923001, 2981.01343443, 70.
        , 7.94625545],
        [ 161.84160321, 4975.87015941, 632.14954806, 527.0203606, 599.81631157
        , 12.40989184, 58.0418942, 1505.01960221, 314.92752409, 1.
        , 338.2751574, 618.00077801, 5.5893879, 2981.06780398, 70.
        , 7.94632857],
        [ 161.93404741, 4975.87224982, 632.15160976, 527.01991281, 599.81622825
        , 12.40952788, 58.04674051, 1504.97514035, 314.92751212, 1.
        , 338.2704948, 617.99351051, 5.58395221, 2981.12250609, 70.
        , 7.94641854],
        [ 161.85375711, 4975.87422687, 632.1536996, 527.01945954, 599.81614523
        , 12.40915701, 58.03655311, 1504.93011524, 314.92750368, 1.
        , 338.26577827, 617.71176209, 5.58404281, 2981.17754715, 70.
        , 7.94652587],
        [ 161.85573731, 4975.87608617, 632.18297035, 527.01900064, 599.81606258
        , 12.40882323, 58.0316865, 1504.88450255, 314.93851325, 1.
        , 338.28587626, 617.43127719, 5.5841343, 2981.23293376, 70.
        , 7.94665113],
        [ 161.94036561, 4975.87788839, 632.18269279, 527.01853597, 599.81598041
        , 12.40848443, 58.0303977, 1504.83827688, 314.93866344, 1.
        , 338.28159063, 617.28530708, 5.58422661, 2981.3592213, 70.
        , 7.94679486],
        [ 162.02805661, 4975.87962952, 632.18241731, 527.02011455, 599.81589881
        , 12.40505846, 58.04245651, 1504.79141171, 314.93881659, 1.
        , 338.27725635, 617.40538166, 5.58431981, 2981.41662113, 70.
        , 7.94695765],
        [ 161.93964021, 4975.75925113, 632.18214485, 527.01970087, 599.81581791
        , 12.40465793, 58.0390514, 1505.45867709, 314.93897278, 1.
        , 338.27286998, 617.38800961, 5.58441391, 2981.4744477, 70.
        , 7.94714007],
        [ 161.85913161, 4975.76046156, 632.18187635, 527.01928206, 599.81573783
        , 12.40945736, 58.03572831, 1505.42573083, 314.93913205, 1.
        , 338.26842791, 617.64575726, 5.5850693, 2981.53271067, 70.
        , 7.94734272],
        [ 161.86181561, 4975.76156233, 632.18156571, 527.01886566, 599.81053568
        , 12.40913163, 58.03104001, 1505.3928835, 314.93929515, 1.
        , 338.2639806, 617.76963131, 5.5851666, 2981.5910439, 70.
        , 7.94756623],
        [ 161.87106601, 4975.76255007, 632.08694857, 527.01845195, 599.8103735
        , 12.40892347, 58.02848081, 1505.36014002, 314.91730542, 1.
        , 338.27204724, 617.76204994, 5.58526441, 2981.64943315, 70.
        , 7.94777597],
        [ 161.87392761, 4975.76342235, 632.08518341, 527.01804111, 599.81021109
        , 12.40862323, 58.03584211, 1505.32749695, 314.91711325, 1.
        , 338.26777896, 617.61638846, 5.58536291, 2981.90074063, 70.
        , 7.94797085],
        [ 161.96361501, 4975.76417677, 632.0833518, 527.01968253, 599.8100485
        , 12.40326644, 58.04122091, 1505.29495017, 314.91691928, 1.
        , 338.2635048, 617.60897993, 5.58546201, 2981.96208661, 70.
        , 7.94814975],
        [ 161.87785011, 4975.7648109, 632.08145089, 527.01912686, 599.8098858
        , 12.40287995, 58.02521961, 1504.87313074, 314.9167235, 1.
        , 338.25922314, 617.46687933, 5.58556181, 2982.02350351, 70.
        , 7.94831149],
        [ 161.78204691, 4975.76244166, 632.07947768, 527.01856914, 599.80972304
        , 12.40249024, 58.03051701, 1504.83111868, 314.91652594, 1.
        , 338.25493218, 617.45923389, 5.58601231, 2982.08498173, 70.
        , 7.94845486],
        [ 161.87421861, 4975.75985252, 632.07742906, 527.01800939, 599.80956027
        , 12.40209716, 58.03132781, 1504.78897557, 314.9163266, 1.
        , 338.25063116, 617.59108232, 5.58605821, 2982.14651127, 70.
        , 7.94857858],
        [ 161.87372001, 4975.75703861, 631.76766859, 527.01744761, 599.80939758
        , 12.40174407, 58.02465461, 1504.74669848, 314.92726792, 1.
        , 338.18957905, 617.45299937, 5.58610291, 2982.20808178, 70.
        , 7.94868136],
        [ 161.87496821, 4975.75399501, 631.7635991, 527.01688381, 599.80939878
        , 12.40138874, 58.03020371, 1504.70428427, 314.92724775, 1.
        , 338.18466369, 617.44298075, 5.58614631, 2982.31341167, 70.
        , 7.94876183],
        [ 161.96151221, 4975.75071677, 631.75946581, 527.01426882, 599.80940554
        , 12.40263598, 58.0335921, 1504.66172955, 314.92722885, 1.
        , 338.17973145, 617.56530235, 5.58618831, 2982.37375178, 70.
        , 7.94881859],
        [ 161.9627368, 4975.94570121, 631.75526566, 527.01372201, 599.80941802
        , 12.40230215, 58.02915581, 1504.97330466, 314.9272113, 1.
        , 338.17478131, 617.42427294, 5.58622901, 2982.43404564, 70.
        , 7.94885018],
        [ 161.88245691, 4975.95131913, 631.75099549, 527.01317443, 599.80943642
        , 12.40196616, 58.03110311, 1504.93747637, 314.92719516, 1.
        , 338.1698122, 617.14111961, 5.58805131, 2982.49427954, 70.
        , 7.9488551, ],
        [ 161.88388141, 4975.9569371, 631.74665195, 527.01262612, 599.83430724
        , 12.40162789, 58.02592241, 1504.90163033, 314.92718051, 1.
        , 338.16482344, 617.2605257, 5.58813621, 2982.55443933, 70.
        , 7.94883177],
        [ 161.88694941, 4975.96255556, 631.46431152, 527.01207708, 599.83474536
        , 12.40129123, 58.02986381, 1504.86576821, 314.93818177, 1.
        , 338.18273498, 617.38123318, 5.58822071, 2982.6145103, 70.
        , 7.94877858],
        [ 161.8900553, 4975.9683077, 631.46051452, 527.01152244, 599.83478808
        , 12.40095208, 58.03450101, 1504.82955236, 314.93835023, 1.
        , 338.1782158, 617.64326389, 5.58830501, 2982.68047438, 70.
        , 7.9487291, ],
        [ 161.89147711, 4975.9742006, 631.45676996, 527.00174066, 599.83482964
        , 12.40236734, 58.02287491, 1504.79296774, 314.93851967, 1.
        , 338.17365743, 617.63902154, 5.58838911, 2982.73987508, 70.
        , 7.94868334],
        [ 161.89341031, 4975.80454046, 631.45307906, 527.00107755, 599.83486996
        , 12.40205162, 58.02785191, 1504.85683068, 314.93869001, 1.
        , 338.16905782, 617.50682129, 5.5884729, 2982.79949187, 70.
        , 7.94864128],
        [ 161.89534071, 4975.80587814, 631.44944299, 527.0004082, 599.83490898
        , 12.40181937, 58.02331991, 1504.82421833, 314.93886113, 1.
        , 338.1644148, 617.50302318, 5.58621091, 2982.85932519, 70.
        , 7.94860293],
        [ 161.98531811, 4975.80723527, 631.44586295, 526.99973238, 599.83007981
        , 12.40150101, 58.02879121, 1504.79132799, 314.93903294, 1.
        , 338.15972646, 617.6303017, 5.58626671, 2982.91937544, 70.
        , 7.94856826],
        [ 161.81482321, 4975.80861634, 631.31529096, 526.99904985, 599.83014538
        , 12.40118006, 58.02564741, 1504.75814458, 314.93920532, 1.
        , 338.1655594, 617.62221339, 5.5863222, 2982.97964303, 70.
        , 7.94853727],
        [ 161.81533321, 4975.81002605, 631.31369517, 526.99836035, 599.83021301
        , 12.40085572, 58.01668191, 1504.72465236, 314.93937815, 1.
        , 338.16234434, 617.47065181, 5.58637741, 2983.12355639, 70.
        , 7.94850994],
        [ 161.90655521, 4975.81146818, 631.3121713, 526.99125992, 599.83028272
        , 12.39938724, 58.02873201, 1504.69083487, 314.93955727, 1.
        , 338.15912071, 617.45794977, 5.58643231, 2983.18698187, 70.
        , 7.94848625],
        [ 161.90240751, 4975.50445701, 631.31072008, 526.99062706, 599.83035457
        , 12.39903711, 58.02232841, 1504.52533931, 314.93974288, 1.
        , 338.15588715, 617.31264646, 5.58648701, 2983.25070083, 70.
        , 7.94846617],
        [ 161.81389841, 4975.50667971, 631.30934222, 526.98998942, 599.8304286
        , 12.39868297, 58.02353871, 1504.48812866, 314.93993522, 1.
        , 338.15264223, 617.16368151, 5.58611901, 2983.31471549, 70.
        , 7.94844967],
        [ 161.81382851, 4975.50895603, 631.30803841, 526.9893468, 599.8253819
        , 12.39834697, 58.02231961, 1504.45044636, 314.94013451, 1.
        , 338.14938545, 617.01022794, 5.58618701, 2983.37902811, 70.
        , 7.94843672],
        [ 161.81233011, 4975.51128914, 631.38339744, 526.98869896, 599.82551082
        , 12.39800748, 58.01848201, 1504.41227713, 314.92932666, 1.
        , 338.07685135, 616.99002174, 5.58625511, 2983.44364097, 70.
        , 7.94842728],
        [ 161.81220041, 4975.51368236, 631.38383523, 526.98804566, 599.82564509
        , 12.39766551, 58.01717651, 1504.37360524, 314.92936003, 1.
        , 338.07103697, 617.23954234, 5.58632341, 2983.69337884, 70.
        , 7.9484211, ],
        [ 161.81202881, 4975.51613685, 631.38431809, 526.98508131, 599.82578081
        , 12.39411518, 58.01615171, 1504.33441448, 314.92939517, 1.
        , 338.06516985, 617.21864603, 5.58639171, 2983.762082, 70.
        , 7.94841874],
        [ 161.81024791, 4975.6407081, 631.38484547, 526.98446672, 599.82591788
        , 12.39371566, 58.01303081, 1503.8462078, 314.92943214, 1.
        , 338.05924916, 617.19734674, 5.58646011, 2983.83115339, 70.
        , 7.94841953],
        [ 162.01207741, 4975.64581729, 631.3854168, 526.98384718, 599.82605619
        , 12.39331281, 58.02222251, 1503.79709499, 314.92947103, 1.
        , 338.05327405, 617.45252033, 5.58411521, 2983.90059625, 70.
        , 7.94842363],
        [ 162.01497441, 4975.6510769, 631.38603146, 526.98322249, 599.82619561
        , 12.39288441, 58.0275438, 1503.74730723, 314.92951189, 1.
        , 338.0472442, 617.44158925, 5.58413071, 2983.97041385, 70.
        , 7.94843096],
        [ 161.93023051, 4975.65649098, 631.61081791, 526.98259244, 599.82633603
        , 12.39245201, 58.01259891, 1503.69682902, 314.94056915, 1.
        , 338.08115731, 617.56628941, 5.58414541, 2984.04060948, 70.
        , 7.94844146],
        [ 161.84932371, 4975.66206372, 631.61292394, 526.98195679, 599.82647957
        , 12.39201512, 58.01182861, 1503.64564441, 314.9406142, 1.
        , 338.07648556, 617.42348144, 5.58415931, 2984.19487867, 70.
        , 7.94845504],
        [ 161.93760001, 4975.66779934, 631.61504472, 526.98233991, 599.82662627
        , 12.39662866, 58.01681941, 1503.59373704, 314.94066144, 1.
        , 338.07177444, 617.27167492, 5.58417241, 2984.26587987, 70.
        , 7.94847162],
        [ 162.20944591, 4975.71662295, 631.61717907, 526.98172596, 599.82677612
        , 12.39626572, 58.01149591, 1504.05261068, 314.94071097, 1.
        , 338.06702339, 617.25571835, 5.58418481, 2984.33722049, 70.
        , 7.94849111],
        [ 162.12692651, 4975.72197207, 631.61932576, 526.98110567, 599.82692914
        , 12.39589976, 58.02190881, 1504.02580314, 314.94076286, 1.
        , 338.06223185, 617.37289718, 5.58321571, 2984.40890158, 70.
        , 7.94851342],
        [ 162.04318691, 4975.72734731, 631.62148355, 526.98047885, 599.80736198
        , 12.39561783, 58.00869001, 1503.99908533, 314.9408172, 1.
        , 338.05740023, 617.22799711, 5.58320931, 2984.48092415, 70.
        , 7.94853844],
        [ 162.04640441, 4975.73274678, 631.83830276, 526.97984524, 599.80695448
        , 12.39533597, 58.01917361, 1503.97246912, 314.92985973, 1.
        , 338.0676552, 617.21787092, 5.58320191, 2984.55328912, 70.
        , 7.94856605],
        [ 162.03514181, 4975.73816856, 631.83982371, 526.97920461, 599.8065399
        , 12.39505431, 58.00641991, 1503.94596684, 314.92973866, 1.
        , 338.06279297, 617.07271291, 5.58319371, 2984.61714428, 70.
        , 7.94859614],
        [ 161.93691671, 4975.74361085, 631.84132321, 526.98086205, 599.80611814
        , 12.3931193, 58.00583961, 1503.91959124, 314.92961437, 1.
        , 338.0578975, 616.78314012, 5.5831846, 2984.68772525, 70.
        , 7.94862859],
        [ 162.04341451, 4975.72761143, 631.84280014, 526.98024024, 599.80568909
        , 12.39281042, 58.01781841, 1503.61279994, 314.92948675, 1.
        , 338.05296962, 616.90535633, 5.58317451, 2984.75857834, 70.
        , 7.94866325],
        [ 162.04633851, 4975.72732826, 631.84425332, 526.9796114, 599.80525262
        , 12.39250214, 58.01787891, 1503.57634488, 314.92935569, 1.
        , 338.04801018, 616.89748989, 5.58624131, 2984.82970231, 70.
        , 7.94869998],
        [ 161.95562811, 4975.72682336, 631.84568159, 526.97897538, 599.80480862
        , 12.39219476, 58.00311521, 1503.53973919, 314.92922107, 1.
        , 338.04302152, 616.88521234, 5.58624631, 2984.9010958, 70.
        , 7.94873864],
        [ 161.95289391, 4975.72635326, 632.05225789, 526.97833199, 599.81945709
        , 12.39188859, 58.01506221, 1503.50299655, 314.9400971, 1.
        , 338.0132919, 617.0093973, 5.58624971, 2984.97275733, 70.
        , 7.94877905],
        [ 162.03889481, 4975.72592165, 632.05363684, 526.97768107, 599.81949277
        , 12.39158418, 58.01416421, 1503.46613163, 314.94031611, 1.
        , 338.0082497, 617.00499012, 5.58625141, 2985.03583619, 70.
        , 7.94882106],
        [ 161.86480711, 4975.72526046, 632.05499097, 526.98137694, 599.81953102
        , 12.38940461, 58.01063241, 1503.42916006, 314.9405366, 1.
        , 338.00319215, 617.13632173, 5.58625141, 2985.10755835, 70.
        , 7.94886447],
        [ 161.8662566, 4976.15246229, 632.05631937, 526.9807608, 599.81957194
        , 12.38904096, 58.01155981, 1502.47513038, 314.94075854, 1.
        , 337.99811824, 617.26620504, 5.58624961, 2985.17956719, 70.
        , 7.9489091, ],
        [ 161.95201741, 4976.15955998, 632.05762108, 526.98013857, 599.8196156
        , 12.38867764, 58.0076238, 1502.42364441, 314.94098186, 1.
        , 337.9930269, 617.40181635, 5.58304631, 2985.25186222, 70.
        , 7.94895474],
        [ 162.0437084, 4976.16680022, 632.0588951, 526.97951017, 599.80967237
        , 12.38831448, 58.01030371, 1502.37202316, 314.9412065, 1.
        , 337.98791699, 617.26772058, 5.58295501, 2985.32444284, 70.
        , 7.94900119],
        [ 161.95560431, 4976.17418732, 632.14646215, 526.97887552, 599.80955807
        , 12.38795132, 58.00108131, 1502.3202574, 314.93041805, 1.
        , 337.94441018, 617.26591039, 5.58285971, 2985.3973084, 70.
        , 7.94904823],
        [ 162.04101011, 4976.18170811, 632.14651037, 526.97823454, 599.80944325
        , 12.38758805, 58.01080121, 1502.26833729, 314.93046458, 1.
        , 337.9387636, 617.39476656, 5.58276041, 2985.48789214, 70.
        , 7.94909562],
        [ 162.04434371, 4976.18936747, 632.14652686, 526.97963634, 599.80932791
        , 12.38644829, 58.00900101, 1502.21625237, 314.93051222, 1.
        , 337.93309852, 617.24990557, 5.58265691, 2985.56212986, 70.
        , 7.94914311],
        [ 161.94959481, 4976.41689208, 632.14651068, 526.97896148, 599.80921206
        , 12.38605811, 58.00796441, 1502.99464398, 314.93056096, 1.
        , 337.92741539, 617.23964436, 5.58254911, 2985.63669864, 70.
        , 7.94919045],
        [ 161.95136521, 4975.47088885, 632.14646089, 526.9782828, 599.80909567
        , 12.38566739, 58.00940931, 1502.96144794, 314.93061074, 1.
        , 337.92171471, 617.36215651, 5.58391791, 2985.71159855, 70.
        , 7.94923736],
        [ 162.04078841, 4976.44208036, 632.14637648, 526.97760034, 599.80897874
        , 12.38527624, 58.00564351, 1502.92843659, 314.93066154, 1.
        , 337.915997, 617.08771833, 5.58391731, 2985.78682958, 70.
        , 7.94928357],
        [ 162.04385451, 4975.52860991, 632.06172774, 526.97691413, 599.81398421
        , 12.38488475, 58.00686581, 1502.89562218, 314.93071331, 1.
        , 337.95562651, 617.20914794, 5.58391531, 2985.86239158, 70.
        , 7.94932878],
        [ 161.95920791, 4975.52291358, 632.05974016, 526.97622418, 599.81395014
        , 12.38991944, 58.00359321, 1502.86301759, 314.93058544, 1.
        , 337.95021258, 617.20550648, 5.58391171, 2985.89295022, 70.
        , 7.94937268],
        [ 161.95951681, 4975.51709432, 632.05772454, 526.97962891, 599.81391686
        , 12.38961509, 58.00568331, 1502.83063629, 314.93045301, 1.
        , 337.94478015, 617.06434172, 5.58390651, 2985.96727365, 70.
        , 7.94941494],
        [ 161.95839321, 4975.51115112, 632.0556804, 526.97896521, 599.81388437
        , 12.38936192, 58.00248691, 1503.1824776, 314.9303159, 1.
        , 337.93932972, 617.18989844, 5.58389971, 2986.04187451, 70.
        , 7.94945522],
        [ 161.95870061, 4975.50708391, 632.05360721, 526.97829741, 599.81385267
        , 12.38396421, 58.00619401, 1503.14516286, 314.93017399, 1.
        , 337.93386185, 617.04397653, 5.58618661, 2986.11675036, 70.
        , 7.94949317],
        [ 161.96033291, 4975.50293017, 632.05150444, 526.97762553, 599.82201848
        , 12.38354629, 58.00169151, 1503.1077765, 314.93002713, 1.
        , 337.92837716, 616.88990021, 5.586205, 2986.19189855, 70.
        , 7.94952843],
        [ 161.96194791, 4975.76156972, 631.79476163, 526.97694946, 599.82220673
        , 12.38312685, 57.9983448, 1503.07031573, 314.93922458, 1.
        , 337.85361057, 616.59909519, 5.58622201, 2986.2673224, 70.
        , 7.94956059],
        [ 161.87551511, 4975.76155365, 631.78997168, 526.9762691, 599.82240071
        , 12.38270396, 57.99871521, 1503.03277762, 314.93958392, 1.
        , 337.84675095, 616.58003036, 5.58623771, 2986.22446121, 70.
        , 7.94958985],
        [ 161.97070001, 4975.76152302, 631.78513215, 526.97327898, 599.82260054
        , 12.37719212, 57.99844281, 1502.99515927, 314.93995229, 1.
        , 337.83984885, 616.41882982, 5.58625201, 2986.29534187, 70.
        , 7.94961639],
        [ 161.97081611, 4975.76147951, 631.7802433, 526.97251827, 599.82280632
        , 12.37676467, 58.00147161, 1502.37778189, 314.94032988, 1.
        , 337.83290367, 616.39931916, 5.58626501, 2986.36637884, 70.
        , 7.94964043],
        [ 161.97234011, 4975.7614249, 631.77530547, 526.97175419, 599.82301813
        , 12.37633655, 57.99936931, 1502.33687453, 314.94071687, 1.
        , 337.82591483, 616.38745342, 5.58902611, 2986.43757352, 70.
        , 7.94966222],
        [ 161.88625631, 4975.76140823, 631.77031904, 526.97098673, 599.82323608
        , 12.38133093, 57.99426411, 1502.29598528, 314.94111344, 1.
        , 337.81888176, 616.50615374, 5.58907551, 2986.50892748, 70.
        , 7.94968201],
        [ 161.97431231, 4976.05517093, 631.45765125, 526.97021587, 599.82499716
        , 12.3809905, 57.99509731, 1502.25511659, 314.94318474, 1.
        , 337.83497875, 616.3556836, 5.58912491, 2986.5804425, 70.
        , 7.9497001, ],
        [ 162.0583128, 4976.06005834, 631.452569, 526.96944158, 599.82525289
        , 12.38065028, 58.00487471, 1502.21427099, 314.94344567, 1.
        , 337.82916563, 616.20193662, 5.58917451, 2986.60946798, 70.
        , 7.9497161, ],
        [ 162.06131841, 4976.06507577, 631.44747015, 526.95893023, 599.82551279
        , 12.3803103, 58.0067488, 1502.1734511, 314.94371121, 1.
        , 337.82333779, 616.04839491, 5.58922411, 2986.67989513, 70.
        , 7.94973246],
        [ 161.9712538, 4976.0702292, 631.44235546, 526.95802304, 599.82577685
        , 12.37994431, 57.99255681, 1501.92934454, 314.94398141, 1.
        , 337.81749527, 616.16620176, 5.58927391, 2986.75048765, 70.
        , 7.94974744],
        [ 161.97139341, 4976.07227064, 631.43722579, 526.95710994, 599.82604502
        , 12.37957779, 58.00055371, 1501.87944355, 314.94425633, 1.
        , 337.81163812, 616.41939936, 5.58717861, 2986.82124899, 70.
        , 7.94976215],
        [ 161.9728514, 4976.07425345, 631.43208202, 526.95619085, 599.8263173
        , 12.37958298, 57.99890031, 1501.82931187, 314.94453603, 1.
        , 337.80576642, 616.5412769, 5.58716431, 2986.89218292, 70.
        , 7.94977702],
        [ 161.8904216, 4976.32162205, 631.26555219, 526.95526568, 599.82131951
        , 12.37922159, 57.98899581, 1501.77894599, 314.9338062, 1.
        , 337.7918266, 616.38476257, 5.58714831, 2986.96329347, 70.
        , 7.94979251],
        [ 161.89047851, 4976.32750838, 631.26229384, 526.95433435, 599.82109941
        , 12.37885979, 57.99956431, 1501.72834235, 314.93373447, 1.
        , 337.78541826, 616.50982043, 5.5871308, 2987.07267493, 70.
        , 7.94980914],
        [ 161.99025491, 4976.33339692, 631.25904432, 526.94878618, 599.82087501
        , 12.37849762, 57.99326031, 1501.67750284, 314.9336588, 1.
        , 337.77897896, 616.21357546, 5.58711151, 2987.14467822, 70.
        , 7.94982684],
        [ 161.99187421, 4976.33928539, 631.25580291, 526.94791808, 599.82064625
        , 12.37810631, 57.98475421, 1502.00508575, 314.93357911, 1.
        , 337.77250882, 616.19020749, 5.5870906, 2987.21688811, 70.
        , 7.94984559],
        [ 161.9934884, 4976.3480517, 631.25256888, 526.94704661, 599.82041307
        , 12.37771377, 57.99866411, 1501.95833583, 314.93349533, 1.
        , 337.76600802, 616.30745562, 5.58586921, 2987.28930444, 70.
        , 7.94986533],
        [ 161.88514601, 4976.35693979, 631.24934144, 526.94617183, 599.82017543
        , 12.37648611, 57.99366531, 1501.91135415, 314.93340736, 1.
        , 337.75947676, 616.28785616, 5.58586361, 2987.36192705, 70.
        , 7.94988602],
        [ 161.96950541, 4976.37668552, 631.28838412, 526.94529379, 599.82001307
        , 12.37607603, 57.99424211, 1501.86414147, 314.94432948, 1.
        , 337.7911248, 616.13188369, 5.58585701, 2987.43475577, 70.
        , 7.94990763],
        [ 162.14026601, 4976.38599813, 631.28794302, 526.94441254, 599.81984696
        , 12.3810556, 57.99199831, 1501.81669879, 314.94441349, 1.
        , 337.78498652, 616.2429541, 5.58584941, 2987.52415583, 70.
        , 7.94993009],
        [ 161.96193501, 4976.39544029, 631.28752085, 526.94096668, 599.81967703
        , 12.37580016, 57.9829967, 1501.76902742, 314.94449603, 1.
        , 337.77880033, 616.48691603, 5.58584071, 2987.5962968, 70.
        , 7.94995337],
        [ 161.96284291, 4975.45942895, 631.28711675, 526.9401422, 599.81950316
        , 12.37541349, 57.99399771, 1501.57221116, 314.944577, 1.
        , 337.77256557, 616.61111014, 5.58583101, 2987.66858065, 70.
        , 7.94997742],
        [ 162.04644121, 4976.4699933, 631.28672981, 526.93931458, 599.81932527
        , 12.37502622, 57.9949741, 1501.52379923, 314.94465625, 1.
        , 337.76628161, 616.59218127, 5.58564221, 2987.74100491, 70.
        , 7.9500021, ],
        [ 161.96809411, 4975.58446616, 631.28635911, 526.93848384, 599.81914325
        , 12.3746384, 57.98848631, 1501.47520366, 314.94473364, 1.
        , 337.75994783, 616.44961569, 5.58563431, 2987.81356704, 70.
        , 7.95002765],
        [ 161.97062301, 4976.50848355, 631.47888281, 526.93765002, 599.81750409
        , 12.37424974, 57.98593051, 1501.42642919, 314.94480903, 1.
        , 337.75186663, 616.57900115, 5.58562531, 2987.88626439, 70.
        , 7.95005373],
        [ 161.88888141, 4976.54763614, 631.48044815, 526.93681316, 599.81737283
        , 12.37386027, 57.98997551, 1501.37748086, 314.94506284, 1.
        , 337.74653901, 616.5722631, 5.585615, 2987.81557962, 70.
        , 7.95008039],
        [ 161.97793301, 4975.61900154, 631.48203196, 526.94058393, 599.81723722
        , 12.3744906, 58.00099221, 1501.32836407, 314.94532027, 1.
        , 337.74121937, 616.5613241, 5.58560331, 2987.88315668, 70.
        , 7.95010759],
        [ 161.98065281, 4975.62068785, 631.48363383, 526.93985447, 599.81709716
        , 12.37416889, 57.98612731, 1501.55794118, 314.94558133, 1.
        , 337.73590871, 616.68348891, 5.58559041, 2987.95071529, 70.
        , 7.95013528],
        [ 161.98344631, 4975.62035965, 631.48525331, 526.93912312, 599.81695256
        , 12.37384845, 57.98655071, 1501.52042861, 314.94584608, 1.
        , 337.73060804, 616.67506501, 5.58415931, 2988.01824909, 70.
        , 7.95016341],
        [ 161.98297691, 4976.54272183, 631.48688998, 526.93838994, 599.82653693
        , 12.37352936, 57.98767891, 1501.48310621, 314.94611452, 1.
        , 337.72531844, 616.52604935, 5.58416011, 2988.08575159, 70.
        , 7.95019193],
        [ 161.98245241, 4975.63455944, 631.72112536, 526.93765498, 599.82654252
        , 12.37321207, 57.98560061, 1501.44598816, 314.93537235, 1.
        , 337.7511824, 616.24515935, 5.5841606, 2988.15321615, 70.
        , 7.9502201, ],
        [ 161.89224801, 4975.63419991, 631.72293368, 526.9369183, 599.82654591
        , 12.37289666, 57.98171601, 1501.40908913, 314.93528715, 1.
        , 337.74577298, 616.22350148, 5.58416081, 2988.18388276, 70.
        , 7.95024996],
        [ 161.80938361, 4975.63374312, 631.72473712, 526.93899759, 599.826547
        , 12.3719989, 57.98151051, 1501.37242429, 314.93519983, 1.
        , 337.74035146, 616.33548951, 5.58416081, 2988.24927627, 70.
        , 7.95027939],
        [ 161.89635701, 4975.63318483, 631.72653536, 526.93828706, 599.8265457
        , 12.37159514, 57.98150801, 1500.77964686, 314.93511029, 1.
        , 337.7349183, 616.59114631, 5.58416061, 2988.31457785, 70.
        , 7.95030903],
        [ 161.97325561, 4975.63181699, 631.72832808, 526.93757494, 599.82654192
        , 12.3711907, 57.98330571, 1500.72599172, 314.93501848, 1.
        , 337.72947395, 616.58025168, 5.58229121, 2988.37978043, 70.
        , 7.95033883],
        [ 161.97073641, 4975.60887867, 631.730115, 526.93686133, 599.82166877
        , 12.37078564, 57.99613671, 1500.67189645, 314.93492431, 1.
        , 337.7240189, 616.70506487, 5.58227601, 2988.4448768, 70.
        , 7.95036875],
        [ 161.88213841, 4976.55117645, 631.94526675, 526.93614628, 599.8219033
        , 12.37037855, 57.97473471, 1500.61735237, 314.92368528, 1.
        , 337.70764485, 616.69649221, 5.58226071, 2988.50985963, 70.
        , 7.95039876],
        [ 161.87944551, 4976.56459333, 631.94702026, 526.93542991, 599.82214837
        , 12.36996941, 57.9740062, 1500.56235045, 314.92358406, 1.
        , 337.70174304, 616.54885691, 5.58224531, 2988.92613186, 70.
        , 7.9504281, ],
        [ 161.87689311, 4976.53388378, 631.94877758, 526.93189467, 599.82240424
        , 12.3699264, 57.99094081, 1500.50688133, 314.92348317, 1.
        , 337.69582419, 616.26644738, 5.58222971, 2988.996771, 70.
        , 7.95045884],
        [ 161.87590911, 4975.62405956, 631.95053913, 526.93109192, 599.82267123
        , 12.36954619, 57.98496531, 1500.33460826, 314.92338266, 1.
        , 337.68988857, 616.11991348, 5.58221411, 2989.06741109, 70.
        , 7.95048885],
        [ 161.77089621, 4975.62236198, 631.95230534, 526.93028621, 599.82294963
        , 12.3691648, 57.96752101, 1500.28086742, 314.92328259, 1.
        , 337.68393642, 616.10703737, 5.58441561, 2989.13804769, 70.
        , 7.95051878],
        [ 161.84975681, 4975.60990704, 631.95407668, 526.9294776, 599.81837294
        , 12.36878223, 57.97988701, 1500.2268432, 314.92318303, 1.
        , 337.67796799, 616.22548282, 5.58438571, 2989.20867631, 70.
        , 7.95054859]]


torch_data = torch.from_numpy(np.array(_data)).float()








