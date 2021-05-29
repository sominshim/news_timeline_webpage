
from module import *

if __name__ == '__main__':
    # 카테고리 정보 : cat_num (0~7) - 숫자형
    # 검색어 정보 : keyword - 문자열

    cat_num = 2
    keyword = 'n번방'
    data_path = 'data/'

    timeline_list = timeline(cat_num, keyword, data_path)
    print(timeline_list)
