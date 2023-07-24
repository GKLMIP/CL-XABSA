
with open('data/rest/cs_nl-en-train.txt') as f:
    data = f.read().strip()
    datas = data.split('\n\n')
    datas = [i.split('\n') for i in datas]


with open('data/rest/smt-nl-train.txt') as f:
    data = f.read().strip()
    datas_cs = data.split('\n\n')
    datas_cs = [i.split('\n') for i in datas_cs]

datas_cs_ = []
for i in datas_cs:
    tmp = []
    for j in i:
        if '\tO\t' not in j:
            tmp.append(j)
    datas_cs_.append(' '.join(tmp))
print(len(datas_cs_))
ids = []
num_ = 0
for num,i in enumerate(datas):
    print(num)
    tmp = []
    for j in i:
        if '\tO\t' not in j:
            tmp.append(j)
    tmp = ' '.join(tmp)
    print(tmp)
    print(datas_cs_[num_])
    print('-------')
    if tmp == datas_cs_[num_]:
        ids.append(num)
        num_ += 1
    if num_ == len(datas_cs_):
        break
print(len(ids))