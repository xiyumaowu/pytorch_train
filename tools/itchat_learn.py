import itchat

itchat.auto_login()
# itchat.run()
# itchat.update_friend()
# for i in itchat.get_contact():
#     print('contact: ' + str(i))
for j in itchat.get_friends():
    print('Friend: ' + str(j))