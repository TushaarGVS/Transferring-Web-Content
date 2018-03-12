import urllib2
import itertools

from bs4 import BeautifulSoup
#from selenium import webdriver


class Element:
    def __init__(self,element,label,
                 containImage, containText, tag, siblingNumber, childrenNumber,
                 level):

        self.containImage = containImage
        self.containText = containText
        self.tag = tag
        self.siblingNumber = siblingNumber
        self.childrenNumber = childrenNumber
        self.level = level
        self.label=label
        self.element = element


def xpath_soup(element):
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:
        """
        @type parent: bs4.element.Tag
        """
        previous = itertools.islice(parent.children,
                                    0, parent.contents.index(child))
        xpath_tag = child.name
        xpath_index = sum(1 for i in previous if i.name == xpath_tag) + 1
        components.append('%s[%d]' % (xpath_tag, xpath_index))
        child = parent
    components.reverse()
    return '/%s' % '/'.join(components)


class WebPage:
    def __init__(self, elementList):
        self.elementList = elementList

    @staticmethod
    def build_from_url(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'lxml')
        soup = soup.body

        def walker(soup, level):
            nodes = []
            if soup.name is not None:
                for child in soup.children:
                    if child.name is not None:
                        nodes.append({'num_siblings': len(list(soup.children))-1,
                                      'num_children': len(list(child.children)),
                                      'level': level+1,
                                      'tag': child.name,
                                      'containImage': child.name == 'img',
                                      'containText': child.text != '',
                                      'element':soup,
                                      })
                        nodes = nodes + walker(child, level+1)
            return nodes

        nodes = walker(soup, 0)


        element_list = []

        for node in nodes:
            #print(node)

            element_list.append(Element(element=node['element'],
                                        label='',
                                        containImage=node['containImage'],
                                        containText=node['containText'],
                                        tag=node['tag'],
                                        siblingNumber=node['num_siblings'],
                                        childrenNumber=node['num_children'],
                                        level=node['level']
                                        )
                                )
        return WebPage(element_list)
