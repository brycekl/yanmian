import pandas as pd

spacing = pd.read_excel('./spacing_outputs.xlsx')
spacing = spacing.to_dict('list')

result_pre = pd.read_excel('./result_pre_500.xlsx')
result_gt = pd.read_excel('./result_gt_500.xlsx')
del result_pre['Unnamed: 0']
del result_gt['Unnamed: 0']

result_pre = result_pre.to_dict('list')
result_gt = result_gt.to_dict('list')

for i in range(500):
    file_name, spa = spacing['name'][i].split('.jpg')[0].split('.JPG')[0], spacing['spacing'][i]
    ind = result_pre['name'].index(file_name)
    assert file_name == result_pre['name'][ind] and file_name == result_gt['name'][ind]
    result_pre['PL'][ind] = result_pre['PL'][ind] * spa * 10
    result_pre['FS'][ind] = result_pre['FS'][ind] * spa * 10
    result_gt['PL'][ind] = result_gt['PL'][ind] * spa * 10
    result_gt['FS'][ind] = result_gt['FS'][ind] * spa * 10

writer = pd.ExcelWriter('./result_pre_500_.xlsx')
df = pd.DataFrame(result_pre)
df.to_excel(writer)
writer.close()
writer = pd.ExcelWriter('./result_gt_500_.xlsx')
df = pd.DataFrame(result_gt)
df.to_excel(writer)
writer.close()


