from sys import argv
import lxml.html as lh

html = lh.parse(argv[1])

root = html.getroot()
rootiter = html.getiterator()

for i in rootiter:
    print(html.getpath(i))
