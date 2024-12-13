from sys import argv

options = ('hello', 'goodbye', 'yes', 'no')

if argv[1] not in options:
    raise Exception('Invalid option')

print(argv[1])
