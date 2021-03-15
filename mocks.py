
epochs = [

    'Outputs/Epochs/Andrews_Model1_128_256_10_1600.pth',   # NOPE   # Want Want
    'Outputs/Epochs/Andrews_Model1_128_256_10_3200.pth',   # NOPE   # Maybe Want

    'Outputs/Epochs/Andrews_Model1_128_256_10_6400.pth',   # YES_   # Nope Want
    'Outputs/Epochs/Andrews_Model1_128_256_10_12800.pth',  # MAYB   # Maybe

    'Outputs/Epochs/Andrews_Model1_128_256_50_1600.pth',   # MAYB   # Nope      # performed bad in features so I'll take the one above it
    'Outputs/Epochs/Andrews_Model1_128_256_50_3200.pth',   # EEHH   # Maybe Want

    'Outputs/Epochs/Andrews_Model1_128_256_50_6400.pth',   # YES_   # Want Want
    'Outputs/Epochs/Andrews_Model1_128_256_50_12800.pth',  # YES_   # Maybe Want

    'Outputs/Epochs/Andrews_Model1_128_256_100_1600.pth',  # NOPE     # Nope Maybe Want
    'Outputs/Epochs/Andrews_Model1_128_256_100_3200.pth',  # NOPE     # Nope
    'Outputs/Epochs/Andrews_Model1_128_256_100_6400.pth',  # EEHH     # Nope

    'Outputs/Epochs/Andrews_Model1_128_256_100_12800.pth', # YES_     # Nope Want

# 0	   1	2	3	4	5	6	7	8	9	10	11
# 71   76	94	85	86	81	95	92	66	51	81	93	length:12
]

features = [
    # 'Outputs/Features/Andrews_Model1_F_CycAmps_10_800.pth',     # NOPE   # NO
    'Outputs/Features/Andrews_Model1_F_CycAmps_10_3200.pth',    # MAYB   # MAYBE
    'Outputs/Features/Andrews_Model1_F_CycAmps_10_6400.pth',    # MAYB   # YES
    # 'Outputs/Features/Andrews_Model1_F_CycAmps_50_1600.pth',    # NOPE   # EH
    'Outputs/Features/Andrews_Model1_F_CycAmps_100_12800.pth',  # YES_   # YES

    # 'Outputs/Features/Andrews_Model1_F_Rejects_10_800.pth',     # EEHH   # MAYBE
    # 'Outputs/Features/Andrews_Model1_F_Rejects_10_3200.pth',    # EEHH   # YES
    'Outputs/Features/Andrews_Model1_F_Rejects_10_6400.pth',    # YES_   # YES
    # 'Outputs/Features/Andrews_Model1_F_Rejects_50_1600.pth',    # NOPE   # NAH
    'Outputs/Features/Andrews_Model1_F_Rejects_100_12800.pth',  # MAYB   # YES

# 0	   1	2	3	4	5	6	7	8	9
# 56   85	88	57	94	69	79	94	54	84	length:10
]


models = features
models = epochs