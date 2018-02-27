#!/usr/bin/expect

set timeout 3
spawn ssh ezhonju@10.166.65.56
expect "Password:"
#spawn sleep 1
send "aaaa2222\r"
interact


vim ~/.bashrc
alias sshoss='/home/ezhonju/python/ssh_oss.sh'