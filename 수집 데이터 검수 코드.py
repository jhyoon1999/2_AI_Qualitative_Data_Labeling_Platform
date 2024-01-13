#%% 사용할 라이브러리 불러오기
import pandas as pd
import os
import re

#%% 첫번째) 파일 불러오기

#1. 폴더 위치를 입력하여 해당 폴더 내의 파일명을 읽어드린다.
folder_path = '****'
files_name = os.listdir(folder_path)
print(pd.Series(files_name))

#(1). 확인해야할 것은 데이터와 코드프레임
#데이터는 '2.', 코드프레임은 '3.'이 파일명 앞에 붙는다.
specific_values = ['2.', '3.']
files_name = [element for element in files_name if any(element.startswith(value) for value in specific_values)]
print(pd.Series(files_name))

#(2). 검수할 데이터와 코드프레임 명을 지정하자
dat_file_name = files_name[0]
codes_file_name = files_name[1]
print('검수할 데이터 명:', dat_file_name)
print('검수할 코드프레임 명:', codes_file_name)

#2. 데이터를 불러온다.
#(1). 데이터 경로 지정하기
dat_path = folder_path + '/' + dat_file_name
print(dat_path)

#(2). 데이터에 1개의 시트만 존재하는지 확인하기
dat_xls = pd.ExcelFile(dat_path)
dat_sheet_names = dat_xls.sheet_names

if len(dat_sheet_names) == 1 :
    print('불러오려는 데이터에는 1개의 시트가 존재합니다.')
else :
    print("warning!!!!!!")
    print("데이터의 Sheet를 1개로 줄여주세요")
    print("warning!!!!!!")

#(3). 데이터를 불러들여온다.
dat = pd.read_excel(dat_path)
print(dat.shape)
print(dat.info())

#(4). 데이터의 열 이름을 편집해주자
columns = dat.columns
new_columns = {}

number_to_edit = list(range(10,60))
number_to_edit = ['.' + str(element) for element in number_to_edit]
number_to_edit = tuple(number_to_edit)

for column in columns:
 if column.endswith(('.1', '.2', '.3', '.4','.5', '.6','.7','.8','.9')):
  new_column = column[:-2]  # Remove the last two characters (.1, .2, .3)
 elif column.endswith(number_to_edit) :
  new_column = column[:-3]
 else:
  new_column = column
 new_columns[column] = new_column
 
dat.rename(columns=new_columns, inplace=True)
print(list(dat.columns))

#(5). 제품명을 추출하자
product_name = list(dat.columns)[0]
print(product_name)

#3. 코드프레임을 불러오자
codes_path = folder_path + '/' + codes_file_name
print(codes_path)

#(1). 코드프레임에는 코드프레임 sheet와 질문지 sheet 2개만 존재해야 한다.
codes_xls = pd.ExcelFile(codes_path)
codes_sheet_names = codes_xls.sheet_names

if len(codes_sheet_names) <= 2 :
    print('불러오려는 코드프레임에는 2개 이하의 시트가 존재합니다.')
else :
    print("warning!!!!!!")
    print("코드프레임 파일의 시트 개수를 확인해주세요!!!")
    print("warning!!!!!!")
    
#(2). 코드프레임을 불러오자
codes = pd.read_excel(codes_path)
#앞서 지정한 제품명이 일치하는 코드프레임만 남긴다.
code = codes[codes['제품'] == product_name]
if code.shape[0] == 0 :
    print("warnings!!!!!")
    print("제품명이 데이터와 코드프레임 모두 일치하는지 확인해주세요!")
    print("warnings!!!!!")
code.head()   

#(3). 동일 문항인데, 번호가 반복되는 잘못된 케이스가 존재하는지 확인해보자.
check_code = code.groupby('문항')['코드'].value_counts()
check_code = check_code[check_code > 1]
if check_code.shape[0] != 0 :
    print("warnings!!!!!!")
    print("동일 문항에서 번호가 반복되는 경우가 존재해요!")
    print(check_code)
    print("warnings!!!!!")


#(5). 코드프레임의 문항과 데이터의 문항은 일치할까?
q_dat = set(dat.columns)
q_dat.discard(product_name)

q_code = set(code['문항'])

if q_dat == q_code :
    print('일치!!')
else : 
    print("warning!!!!!!!!")
    for question1 in q_dat :
        if question1 not in q_code :
            print('코드프레임에 해당 문항이 존재하지 않습니다:', question1)
    for question2 in q_code :
        if question2 not in q_dat :
            print('데이터에 해당 문항이 존재하지 않습니다:', question2)

#%% 두번째) 데이터와 코드프레임의 Sub-Net을 비교해보자.
#1. 혹시 데이터에서 이상한 값이 없을까? 숫자가 아닌 값이 subnet 열에 들어갔는지 확인!
dat_new = dat.copy()
new_column = dat_new.loc[0]
new_column
dat_new.columns = new_column
dat_new = dat_new[1:]
dat_new = dat_new.iloc[:, 1:]

specific_string = "subnet"
subnet_columns_type = dat_new.filter(like = specific_string)
subnet_columns_type.shape

for i in range(subnet_columns_type.shape[1]) :
    subnet_target = subnet_columns_type.iloc[:,i]
    subnet_target
    non_digit_rows = subnet_target.dropna()[~subnet_target.astype(str).str.match(r'^(\d+|NaN)$')]
    if len(non_digit_rows) != 0 :
        print("warning!!!")
        print("subnet 열에서 이상한 값이 발견되었습니다.")
        print('이상 감지 subnet 위치 : ', i+1)
        print('해당 subnet에서 몇번째 행? :', list(non_digit_rows.index))
    else :
        print("문제 없어요!")
    
#2. 데이터에는 있는 번호가 코드프레임에 없는 경우를 찾아보자
for target in list(q_dat) :
    print('##################################')
    print('검수문항 : ', target)
    
    subnet_dat = dat.loc[:, target]
    subnet_new_column = subnet_dat.loc[0]
    subnet_dat.columns = subnet_new_column
    subnet_dat = subnet_dat[1:]
    
    specific_string = 'subnet'
    subnet_dat = subnet_dat.filter(like = specific_string)
    subnet_dat = set(subnet_dat.stack().values) 
    
    subnet_code = code[code['문항'] == target]
    subnet_code = set(subnet_code['코드'])
    for element in subnet_dat :
        if int(element) not in subnet_code:
            print("------------------")
            print('문항 : ', target)
            print("문제발견 : ", element)
            print("------------------")

#%% 세번째) 그 외 검수내용
#1. Net가 없는 Sub-Net이 존재하는가?
notnet = code[code['Net'].isna()]
if notnet.shape[0] != 0 :
    print('Net가 없는 Sub-Net이 존재합니다.')
    print(list(notnet['문항'].unique()))
else :
    print("Net가 모두 존재해요!!")

#2. 없음, 모름, 무응답은 모두 '기타'로 처리되었을까?
target_null = ['없다', '없음', '모름/무응답', '모름', '무응답']
null_code = code[code['Sub-Net'].isin(target_null)]
print(list(null_code['Net'].unique()))

