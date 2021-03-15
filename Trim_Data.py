import sys

import pandas as pd


"""
TODO
- Do I need to grab the lower value of passing or the closest one?? I forgot
- DONE:Look out for `[-11059] No Good Data For Calculation`

TOSETUP:
 - DONE:Nopen is not implemented so do that yourself
 - Name of N or S columns have to be `N or S?` and `CycPmp_Amps`
 - Provide indexs for (  All provided indexes will all be deleted ):
     + index_of_N = `\\RoasterPI\M_G2_Cyc_Feed_Pmp_N_Amps:DCS`
     + index_of_S = `\\RoasterPI\M_G2_Cyc_Feed_Pmp_S_Amps:DCS`
     + indexs_of_Cycs = list(range(`22`, `33`+1))
 - Finds largest of N or S and adds it to `CycPmp_Amps` then puts a 1 or 2 in `N or S?` depending if its N or S
 - Sums the Cyclone Columns (indexs_of_Cycs) and puts it sum in 'Nopen' column
"""



file = 'data/Collected_6-10_6-25-2020.csv'
index_of_N = 29
index_of_S = 30
indexs_of_Cycs = list(range(17, 28 + 1))
# indexs_of_NorS_&_Amps = 10,11

print("File format must be like this: \n")
print(",5/1/2020 0:10,5/1/2020 0:20,,,,,, \t\t...\n"
      ",\\roasterpi\M_G2_Field_Thick_Feed_%Pass:Man,\\roasterpi\M_G2_SAG_Spd_CO_%:DCS, \t\t...\n"
      ",,,,,,,,,,,\t\t...\n"
      "5/1/2020 0:10,67.19999695,90,308.8522128,4932.6461, \t\t... \n"
      "5/1/2020 0:20,67.19999695,90,314.9070741,4913.5457, \t\t... \n\n"
      )

if len(sys.argv) < 2:
    print(f"No file specified, I'm going to use the hard coded file location\n'{file}'\n\n")
else:
    file = sys.argv[1]

df = pd.read_csv(file, index_col=0, low_memory=False)
df.columns = df.iloc[0]
df = df.iloc[1:]
df.index = list(range(len(df)))
print(df)

# creates array of True or False if they should be kept due to %passing
bols = []
for idx, row in df.iterrows():
    try:
        p = float(row[0])
        nt1 = float(df.iloc[idx + 1][0])
        nt2 = float(df.iloc[idx + 2][0])
        bol = not ((p == nt1) | (p == nt2) | (nt1 != nt2))

        # if I need this row set it to be the next value
        if bol:
            row[0] = nt1
        bols.append(bol)

        # Gets:
        #   index_of_N = `\\RoasterPI\M_G2_Cyc_Feed_Pmp_N_Amps:DCS`
        #   index_of_S = `\\RoasterPI\M_G2_Cyc_Feed_Pmp_S_Amps:DCS`
        # And then assigns the 'N or S?' and 'CycPmp_Amps' columns
        N = float(row[index_of_N])
        S = float(row[index_of_S])

        if N > S:
            row['N or S?'] = 1
            row['CycPmp_Amps'] = N
        else:
            row['N or S?'] = 2
            row['CycPmp_Amps'] = S

        # Sums the Cyclone Columns and puts it sum in 'Nopen' column
        sum = 0
        for i in indexs_of_Cycs:
            sum += float(row[i])
        row['Nopen'] = sum

    except Exception as e:
        pass

print("END OF FILE, MOST LIKELY MISSING LAST TWO LINES")
print("That's why I remove the last two of the df :: df = df.iloc[:-2]\n")

# removes index_of_N = 34, index_of_S = 35
print(df.columns[index_of_N], df.columns[index_of_S], '\n')
df = df.drop(df.columns[index_of_S], axis=1)
df = df.drop(df.columns[index_of_N], axis=1)

# removes Cyclone columns
indexs_of_Cycs.reverse()
for i in indexs_of_Cycs:
    print(df.columns[i])
    df = df.drop(df.columns[i], axis=1)
print()

# Adds the bols array to the df and removes all False
# Then removes that column
df = df.iloc[:-2]
df['bols'] = bols
df = df.loc[(df['bols'] == True)]
df.columns.name = None
df = df.drop(['bols'], axis=1)

# looks through every element and checks if it is NaN or str
# and then adds it to remove = [] and then I remove it
remove = []
for i, r in df.iterrows():
    for v in r:
        try:
            if pd.isna(float(v)):
                remove.append(i)
        except:
            remove.append(i)

# duplicates, and reverse or we're taking from the bottom first
rem = list(dict.fromkeys(remove))
rem.reverse()
for i in rem:
    df = df.drop(i)

# saving the data to a file named the same with ` - CLEANED_DATA to the end
print(df)
df.to_csv(f"{file[:-4]} - CLEANED_DATA.csv", index=False)
