from email.mime.text import MIMEText
import smtplib
from email.header import Header
mailInfo = {
    "from": '357594634@qq.com',
    "to": 'junliang_zhong@163.com',
    "hostname": 'smtp.qq.com',
    "username": '357594634@qq.com',
    "password": 'sycvoaenwndabibf',
    "mailsubject": 'python stmp',
    "mailtext": "hello, here is Python...",
    "mailencoding": 'utf-8'
}

smtp = smtplib.SMTP_SSL(mailInfo['hostname'])
smtp.set_debuglevel(1)
smtp.ehlo(mailInfo['hostname'])
smtp.login(mailInfo['username'], mailInfo['password'])
msg = MIMEText(mailInfo["mailtext"], "text", mailInfo["mailencoding"])
msg["Subject"] = Header(mailInfo["mailsubject"], mailInfo["mailencoding"])
msg["from"] = mailInfo["from"]
msg["to"] = mailInfo["to"]
smtp.sendmail(mailInfo["from"], mailInfo["to"], msg.as_string())
smtp.quit()
