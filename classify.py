# http://jkorpela.fi/HTML3.2/3.8.html
# html, head are not content nodes
def get_index(tag):
    tag_index = {'title': 0,
                 'isindex': 1,
                 'base': 2,
                 'meta': 3,
                 'link': 4,
                 'script': 5,
                 'style': 6,
                 'body': 100,
                 'h1': 101,
                 'h2': 102,
                 'h3': 103,
                 'h4': 104,
                 'h5': 105,
                 'h6': 106,
                 'address': 107,
                 'p': 108,
                 'ul': 109,
                 'ol': 110,
                 'dl': 111,
                 'pre': 112,
                 'div': 113,
                 'center': 114,
                 'blockquote': 115,
                 'form': 116,
                 'hr': 117,
                 'table': 118,
                 'em': 200,
                 'strong': 201,
                 'dfn': 202,
                 'code': 203,
                 'samp': 204,
                 'kbd': 205,
                 'var': 206,
                 'cite': 207,
                 'tt': 250,
                 'i': 251,
                 'b': 252,
                 'u': 253,
                 'strike': 254,
                 'big': 255,
                 'small': 256,
                 'sub': 257,
                 'sup': 258,
                 'a': 300,
                 'img': 301,
                 'applet': 302,
                 'font': 303,
                 'basefont': 304,
                 'br': 305,
                 'script': 306,
                 'map': 307,
                 'input': 350,
                 'select': 351,
                 'textarea': 352
                }
    if tag in tag_index:
        return tag_index[tag]
    else:
        return -1


def get_X_from_elements(element_list):
    X = []
    elem_list2 = []
    for element in element_list:
        element_params = []
        if get_index(element.tag) != -1:
            element_params.extend([element.containImage, element.containText, get_index(element.tag), element.siblingNumber, element.childrenNumber, element.level])
            X.append(element_params)
            elem_list2.append(element)
    return X, elem_list2
