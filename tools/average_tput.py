import re
import sys
import getopt

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:")  # ':' means -f must have a parameter
    except getopt.GetoptError:
        print('Usage:-f <logfile or folder>')
        sys.exit(2)
    if len(opts) == 0 :
        print('Usage:-f <logfile or folder>')
    for opt, arg in opts:
        if opt == '-h':
            print('Usage:-f <logfile or folder>')
        elif opt == '-f':
            filename = arg
    # print(filename)
            re_set = re.compile('K?Bytes(.*?)K?bits/sec')
            txt = open(filename)
            counter = 0
            total = 0.0
            for line in txt.readlines():
                if re.findall(re_set, line):
                    counter += 1
                    total = total + float(re.findall(re_set, line)[0])
                    print(float( re.findall(re_set, line)[0]))
            print('average tput: ', total/float(counter ))

if __name__ == '__main__':
    main(sys.argv[1:])