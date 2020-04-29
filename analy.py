

mss_per = []
with open("data/train_dict/missing_pre.txt","r",encoding="utf-8") as f:
    for line in f:
        mss_per.append(line.strip())

dict_per = []
with open("data/train_dict/PER.txt","r",encoding="utf-8") as f:
    for line in f:
        dict_per.append(line.strip())

i = 0
for m in mss_per:
    if m not in dict_per:
        print(i, m)
        i += 1
    else:
        print("in but wrong", m)