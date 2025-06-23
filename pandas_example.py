import pandas as pd
import numpy as np
import random
import networkx as nx

# # SaveData函数的测试示例
# def SaveData(data, rowname, colname, allname, filename):
#     df = pd.DataFrame(data, columns=colname)
#     df.insert(0, allname, rowname)
#     df.to_csv(filename, index=False)
#
# data = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
# # data = {'a': 1, 'b': [1, 4, 7], 'c': [2, 5, 8], 'd': 4} # colname = data.keys() # col需要与键名相同
# rowname = ['row1', 'row2', 'row3']
# colname = ['col1', 'col2', 'col3', 'col4']
# allname = 'r\c'
# filename = 'test.csv'
#
# SaveData(data, rowname, colname, allname, filename)


# Excel保存代码测试示例
a = np.random.randint(1,10,(3,2))
b = np.random.randint(10,100,(3,2))
print(a)
print(b)

df1 = pd.DataFrame(a,columns=['foo', 'bar'],index=['a', 'b', 'c'])
df2 = pd.DataFrame(b,columns=['foo', 'bar'],index=['a', 'b', 'c'])
df1 = df1.append({'bar': "<[0,1];[3,6];[8,8]>", 'foo':"<[0,4,0]; [4,8,1]>"}, ignore_index=True)
print(df1)
print(df2)

with pd.ExcelWriter("excel_writer.xlsx") as writer:
    df1.to_excel(excel_writer=writer,sheet_name="df1",index=None)
    df2.to_excel(excel_writer=writer,sheet_name="df2",index=None)

