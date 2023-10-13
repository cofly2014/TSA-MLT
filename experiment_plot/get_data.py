import json
'''
def get_data():
     myfile = open('attention_score_data.txt', 'r')
     #data = myfile.read().replace('\n', '')
     #data = data.replace(" ","")
     data = myfile.load(myfile)
     return data
'''

def get_data():
    with open("attention_score_data_bk.txt", "r") as f:
        return json.load(f)