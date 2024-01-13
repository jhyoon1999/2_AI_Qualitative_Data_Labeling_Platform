import pandas as pd
import mysql.connector

#%%0. DB로부터 데이터 추출
def fetch_table_data(cursor, table_name):
    query = f'SELECT * FROM {table_name}'
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    return pd.DataFrame(result, columns=columns)

db_config = {
    'host': '****',
    'user': '****',
    'password': '****',
    'database': '****',
    'port': 30575
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

data_table_name = "food_project"
data = fetch_table_data(cursor, data_table_name)

data = data[['response']]
data.columns = ['text']

#%%1.불필요한 문자 제거
import re
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    
def cleaning(texts, punct, mapping):
    for text in texts:
        for p in mapping:	# 특수부호 mapping
            text = text.replace(p, mapping[p])
        
        for p in punct:		# 특수부호 제거
            text = text.replace(p, '')
		
        text = re.sub(r'<[^>]+>', '', text) 	  # remove Html tags
        text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', text) 	  # remove e-mail
        text = re.sub(r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', ' ', text)	  # remove URL  
        text = re.sub(r'\s+', ' ', text)		  # Remove extra space
        text = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', text)	# Remove 한글 자음, 모음
        text = re.sub('[^\w\s\n]', '', text)	  # Remove 특수기호
    
    return texts

cleaning_sentences = cleaning(data['text'], punct, punct_mapping)
len(cleaning_sentences)

data['text'] = cleaning_sentences
data.head()

#%%2. 띄어쓰기 검사
from pykospacing import Spacing

# 띄어쓰기 오류 교정 함수 정의
def correct_spacing(text):
    spacing = Spacing()
    corrected_text = spacing(text)
    return corrected_text

# 'text' 열에 띄어쓰기 오류 교정 및 명사 추출하여 새로운 열 추가
data['corrected_text'] = data['text'].apply(correct_spacing)

#%%3. 맞춤법 검사
from hanspell import spell_checker

def spell_check_korean(text):
    spelled_sent = spell_checker.check(text)
    return spelled_sent.checked

data['spell_check_text'] = data['corrected_text'].apply(spell_check_korean)
data.head()

#%%4. 품사 태깅
#문서 토픽이 유사한 텍스트끼리 묶을 것이기 때문에 명사를 추출한다.
from konlpy.tag import Okt

# 명사 추출 함수 정의
def tokenize(text):
    okt = Okt()
    nouns = okt.nouns(text)
    return ' '.join(nouns)

#명사 태깅
data['tokenized_text'] = data['spell_check_text'].apply(tokenize)
data.head()

data.to_excel("preprocessing_data.xlsx", index=False)
















































































