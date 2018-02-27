import requests
import itchat
from itchat.content import *

# KEY = '29cac6b582b441e887b0e45e873e2ff4' #Tulling Key
KEY = 'hV4CfmCYcLJPFG0'
def get_response(msg):
    # apiurl = 'http://www.tuling123.com/openapi/api'
    apiurl = 'https://convoecom.com/bot/messenger-webhook/CUXb2t1OE4vmpr1apf2Q'
    data = {
        'key': KEY,
        'info': msg,
        'userid': 'wechat-robot'
    }
    try:
        r = requests.post(apiurl, data=data).json() #get tulling robot answer
        print(r.get('text'))
        return r.get('text')
    except Exception as e:
        print(e)

@itchat.msg_register(itchat.content.TEXT)
def tulling_reply(msg):
    try:
        if 'Robot on' == msg['Text']:  #command to on Robot
            TXT = open('isRobot.txt', 'w')
            TXT.write('1')
            TXT.close()
        if 'Robot off' == msg['Text']:  # command to off Robot
            TXT = open('isRobot.txt', 'w')
            TXT.write('0')
            TXT.close()
    except Exception as e:
        print(e)
    try:
        f = open('isRobot.txt', 'r')
        isRobot = f.read()
        f.close()
        if '1' in isRobot:
            defaultReply = 'Received: ' + msg['Text']
            tulling_replys = get_response(msg['Text'])
            return  tulling_replys or defaultReply
        else:
            return
    except Exception as e:
        print(e)

def main():
    itchat.auto_login(True)
    itchat.run()

if __name__ == '__main__':
    main()